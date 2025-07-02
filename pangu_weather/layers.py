import collections
import logging
import math

import numpy as np
import torch
import torch.utils.checkpoint
import timm.layers

# Implementation of PanguWeather in PyTorch based on the official pseudocode given in
# https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py


logger = logging.getLogger(__name__)


def flatten_list(nested_list):
    return [token for sublist in nested_list for token in sublist]


class PatchEmbedding(torch.nn.Module):
    def __init__(self, patch_size, weather_statistics, constant_maps, const_h, dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.conv = torch.nn.Conv1d(in_channels=192, out_channels=dim, kernel_size=1, stride=1)
        self.conv_surface = torch.nn.Conv1d(in_channels=112, out_channels=dim, kernel_size=1, stride=1)

        # weather_statistics = mean and std over the ERA5 training data, used to normalize inputs in the patch embedding
        # passed as 4 tuple: (surface_mean, surface_std, upper_air_mean, upper_air_std)
        # constant_maps = most likely the land mask, soil type, and topography data
        # const_h = an auxiliary mask of shape (1, 1, 1, 13, 721, 1440), unclear what it contains
        self.surface_mean, self.surface_std, self.upper_mean, self.upper_std = [
            torch.nn.parameter.Buffer(tensor, persistent=False) for tensor in weather_statistics]
        self.constant_maps = torch.nn.parameter.Buffer(constant_maps, persistent=False)
        self.const_h = torch.nn.parameter.Buffer(const_h.squeeze(), persistent=False)

        # reshape and reorder weather statistics to expected shape
        # surface: reshape [4] to [1, 4, 1, 1] to match surface input [B, 4, 721, 1440]
        self.surface_mean = self.surface_mean.view(1, 4, 1, 1)
        self.surface_std = self.surface_std.view(1, 4, 1, 1)
        # upper air: flip pressure levels & reshape [13, 1, 1, 5] to [5, 13, 1, 1] to match input [B, 5, 13, 721, 1440]
        self.upper_mean = self.upper_mean.flip(0).permute(3, 0, 1, 2)
        self.upper_std = self.upper_std.flip(0).permute(3, 0, 1, 2)

    @staticmethod
    def pad_to_patch_size(x, patch_size):
        # how much to patch at the end of each dimension
        pad_last = [(patch - true_dim) % patch for true_dim, patch in zip(x.shape[::-1], patch_size[::-1])]
        # build paddings
        padding = [0 for _ in range(2 * len(patch_size))]
        padding[1::2] = pad_last
        return torch.nn.functional.pad(x, padding)  # pad with zeros

    @staticmethod
    def reshape_into_patches(x, patch_size):
        if len(x.shape) != 2 + len(patch_size):
            logger.warning(f'Mismatched shapes in patch embedding: {x.shape=} and {patch_size}.')

        # split last len(patch_size) dimensions into two: remainder patch dimension and (spatial) feature dimension
        # for surface (2D, patch_size=(4, 4)): [B, 7, 724, 1440] -> [B, 7, 181, 4, 360, 4]
        # for upper air (3D, patch_size=(2, 4, 4)): [B, 6, 14, 724, 1440] -> [B, 6, 7, 2, 181, 4, 360, 4]
        split_dims = [[dim // patch, patch] for dim, patch in zip(x.shape[2:], patch_size)]
        split_x = x.view(x.shape[:2] + tuple(flatten_list(split_dims)))

        # reorder to move patch dimensions to end
        # for surface (2D): [B, 7, 181, 4, 360, 4] -> [B, 7, 4, 4, 181, 360]
        # for upper air (3D): [B, 6, 7, 2, 181, 4, 360, 4] -> [B, 6, 2, 4, 4, 7, 181, 360]
        # unaffected dimensions (batch and input features) + (spatial) feature dimensions + patch dimensions
        old_dim_order = tuple(range(len(split_x.shape)))
        new_dim_order = old_dim_order[:2] + old_dim_order[3::2] + old_dim_order[2::2]
        permuted_x = torch.permute(split_x, new_dim_order)

        # aggregate feature dimensions
        # for surface (2D): [B, 7, 4, 4, 181, 360] -> [B, 112, 65160]
        # for upper air (3D): [B, 6, 2, 4, 4, 7, 181, 360] -> [B, 192, 456120]
        batch_size = permuted_x.shape[0]
        feature_dim = int(np.prod(permuted_x.shape[1:2 + len(patch_size)]))
        return permuted_x.reshape(batch_size, feature_dim, -1)

    def forward(self, upper_air_data, surface_data):
        # -------------- embedding of surface variables --------------
        # normalize surface level input, input_surface.shape = [B, 4, 721, 1440],
        normalized_input_surface = (surface_data - self.surface_mean) / self.surface_std

        # pad surface level input and append constant masks as features
        # [B, 4, 721, 1440] -- padding --> [B, 4, 724, 1440] -- append masks --> [B, 7, 724, 1440]
        batch_size = surface_data.shape[0]
        padded_input_surface = self.pad_to_patch_size(normalized_input_surface, self.patch_size[1:])
        input_surface_with_masks = torch.cat(
            (padded_input_surface, self.constant_maps.expand(batch_size, -1, -1, -1)), dim=1)

        # patch embedding: [B, 7, 724, 1440] -- patches --> [B, 112, 65160] -- conv --> [B, self.dim, 65160]
        patched_surface = self.reshape_into_patches(input_surface_with_masks, self.patch_size[1:])
        patched_surface = self.conv_surface(patched_surface)

        # -------------- embedding of upper air variables --------------
        # normalize upper air input, input.shape = [B, 5, 13, 721, 1440]
        normalized_input_upper = (upper_air_data - self.upper_mean) / self.upper_std

        # append const_h as features, then pad upper air input
        # [B, 5, 13, 721, 1440] -- append const_h --> [B, 6, 13, 721, 1440] -- padding --> [B, 6, 14, 724, 1440]
        # reshape const_h from [1, 1, 1, 13, 721, 1440] to [B, 1, 13, 721, 1440]), broadcasting along batch dimension
        const_h_expanded = self.const_h.expand(batch_size, 1, -1, -1, -1)
        input_upper_with_const_h = torch.cat((normalized_input_upper, const_h_expanded), dim=1)
        padded_input_upper = self.pad_to_patch_size(input_upper_with_const_h, self.patch_size)

        # patch embedding: [B, 6, 14, 724, 1440] -- patches --> [1, 192, 456120] -- conv --> [1, self.dim, 456120]
        patched_upper = self.reshape_into_patches(padded_input_upper, self.patch_size)
        patched_upper = self.conv(patched_upper)

        # -------------- combine surface and upper air and reshape into (B, spatial, C) --------------
        # [B, self.dim, 65160] + [1, self.dim, 456120] -> [B, self.dim, 521280] -- permute --> [B, 521280, self.dim]
        x = torch.cat((patched_surface, patched_upper), dim=2)
        x = torch.permute(x, (0, 2, 1))
        return x


def pad_to_shape(x, shape):
    # pad x in each dimension, so it is evenly divisible by the given shape
    # the shape should have the same number of dimensions as x, set a dimension to 1 in shape to avoid padding it
    pad_last = [(patch - true_dim) % patch for true_dim, patch in zip(x.shape, shape)]
    # build paddings
    padding = [0 for _ in range(2 * len(shape))]  # only pad last of each dimension, set first to 0
    padding[1::2] = pad_last[::-1]  # reverse as torch.nn.functional.pad expects the padding to start from last dim
    return torch.nn.functional.pad(x, padding)  # pad with zeros


class DownSample(torch.nn.Module):
    def __init__(self, dim=192, input_shape_3d=(8, 181, 360)):
        super().__init__()
        self.dim = dim
        self.shape_3d = input_shape_3d
        self.patch_size = (2, 2)
        self.linear = torch.nn.Linear(in_features=4 * self.dim, out_features=2 * self.dim, bias=False)
        self.norm = torch.nn.LayerNorm(4 * self.dim)

    def forward(self, x):
        # Reshape to 3D for downsampling: [B, 521280, 192] -> [B, 8, 181, 360, 192]
        x = x.view(x.shape[0], *self.shape_3d, x.shape[-1])

        # Pad to patch size: [B, 8, 181, 360, 192] -> [B, 8, 182, 360, 192]
        x = pad_to_shape(x, (1, 1, *self.patch_size, 1))

        # Reshape to patches: [B, 8, 182, 360, 192] -> [B, 8, 91, 2, 180, 2, 192]
        split_dims = [[dim // patch, patch] for dim, patch in zip(x.shape[2:], self.patch_size)]
        x = x.view(x.shape[:2] + tuple(flatten_list(split_dims)) + x.shape[-1:])

        # Reorder to move patch dimensions together: [B, 8, 91, 2, 180, 2, 192] -> [B, 8, 91, 180, 2, 2, 192]
        x = x.permute((0, 1, 2, 4, 3, 5, 6))

        # Aggregate feature dimensions: [B, 8, 91, 180, 2, 2, 192] -> [B, 131040, 768]
        x = x.reshape(x.shape[0], -1, 4 * self.dim)

        # Apply layer normalization and linear layer: [B, 131040, 768] -- linear --> [B, 131040, 384]
        x = self.norm(x)
        x = self.linear(x)
        return x


class UpSample(torch.nn.Module):
    def __init__(self, input_dim=384, output_dim=192, input_shape_3d=(8, 91, 180), output_shape_3d=(8, 181, 360)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_shape_3d = input_shape_3d
        self.output_shape_3d = output_shape_3d
        self.patch_size = (2, 2)

        self.linear1 = torch.nn.Linear(self.input_dim, 4 * self.output_dim, bias=False)
        self.linear2 = torch.nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.norm = torch.nn.LayerNorm(self.output_dim)

    def forward(self, x):
        # Apply first linear layer: [B, 131040, 384] -- linear --> [B, 131040, 768]
        x = self.linear1(x)

        # Reshape to 3D for upsampling: [B, 131040, 768] -> [B, 8, 91, 180, 2, 2, 192]
        x = x.view(x.shape[0], *self.input_shape_3d, *self.patch_size, -1)
        # Reorder to move patch dims to spatial dims: [B, 8, 91, 180, 2, 2, 192] -> [B, 8, 91, 2, 180, 2, 192]
        x = x.permute((0, 1, 2, 4, 3, 5, 6))
        # Merge patch and spatial dimensions: [B, 8, 91, 2, 180, 2, 192] -> [B, 8, 182, 360, 192]
        up_scaled_spatial = tuple(patch * dim for patch, dim in zip((1, ) + self.patch_size, self.input_shape_3d))
        x = x.reshape(x.shape[:1] + up_scaled_spatial + x.shape[-1:])
        # Crop spatial dimensions to output dimensions: [B, 8, 182, 360, 192] -> [B, 8, 181, 360, 192]
        z_slice, h_slice, w_slice = [slice(0, dim) for dim in self.output_shape_3d]
        x = x[:, z_slice, h_slice, w_slice, :]

        # Aggregate feature dimensions: [B, 8, 181, 360, 192] -> [B, 521280, 192]
        x = x.reshape(x.shape[0], -1, self.output_dim)

        # Apply layer normalization and second linear layer: [B, 521280, 192] --norm & linear --> [B, 521280, 192]
        x = self.norm(x)
        x = self.linear2(x)
        return x


class EarthSpecificLayer(torch.nn.Module):
    def __init__(self, depth, dim, drop_path_ratio_list, num_heads, zhw, checkpoint=True, reproduce_mask=False):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.checkpoint = checkpoint
        assert len(drop_path_ratio_list) == self.depth

        # initialize blocks, roll input every odd block, i.e. roll=bool(i % 2)
        blocks = collections.OrderedDict([
            (f'EarthSpecificBlock{i}',
             EarthSpecificBlock(dim, drop_path, i % 2, zhw, num_heads, reproduce_mask=reproduce_mask))
            for i, drop_path in enumerate(drop_path_ratio_list)])
        self.blocks = torch.nn.Sequential(blocks)

    def forward(self, x):
        if self.training and self.checkpoint:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x)
        else:
            x = self.blocks(x)
        return x


class EarthSpecificBlock(torch.nn.Module):
    """
    3D transformer block with Earth-Specific bias and window attention.
    Swin-Transformer's 2D window attention is extended to 3D and the relative position bias is replaced with an
    earth-specific bias.
    """
    def __init__(self, dim, drop_path_ratio, roll, zhw, num_heads=6, window_size=(2, 6, 12), reproduce_mask=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_dim = int(np.prod(self.window_size))

        self.roll = roll
        self.zhw = zhw  # the Z, H, W dimensions [8, 181, 360] or [8, 96, 180]
        # number of Z, H windows (124 for [8, 181, 360] or 64 for [8, 91, 180]), round up size zhw is not yet padded
        window_counts = [math.ceil(dim / size) for dim, size in zip(self.zhw, self.window_size)]
        self.type_of_windows = window_counts[0] * window_counts[1]

        self.drop_path = timm.layers.DropPath(drop_path_ratio) if drop_path_ratio > 0. else torch.nn.Identity()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.linear = MLP(dim, 0)
        self.attention = EarthAttention3D(dim, num_heads, 0, window_size, self.type_of_windows)

        self.reproduce_mask = reproduce_mask

    def window_partition(self, x):
        # Example shapes for x = [B, 8, 186, 360, C] and window_size = [2, 6, 12]
        # partition into windows: [B, 8, 186, 360, C] -> [B, 4, 2, 31, 6, 30, 12, C]
        batch_size = x.shape[0]
        split_dims = flatten_list([[dim // size, size] for dim, size in zip(x.shape[1:], self.window_size)])
        x = x.view(batch_size, *split_dims, -1)

        # reorder and merge dimensions
        # TODO: should we merge the batch dimension with the second dimension (B * 30 instead of B, 30)?
        # [B, 4, 2, 31, 6, 30, 12, C] -- permute --> [B, 30, 4, 31, 2, 6, 12, C] -- reshape --> [B, 30, 124, 144, C]
        x = x.permute(0, 5, 1, 3, 2, 4, 6, 7)
        x = x.reshape(batch_size, x.shape[1], x.shape[2] * x.shape[3], self.window_dim, x.shape[-1])
        return x

    def window_reverse(self, x, original_shape):
        # Example shapes for x = [B, 30, 124, 144, C], original_shape = [B, 8, 186, 360, C], window_size = [2, 6, 12]
        # split merged dimensions: [B, 30, 124, 144, C] -> [B, 30, 4, 31, 2, 6, 12, C]
        window_counts = [dim // size for dim, size in zip(original_shape[1:], self.window_size)]
        x = x.view(-1, window_counts[-1], *window_counts[:2], *self.window_size, x.shape[-1])
        # reorder: [B, 30, 4, 31, 2, 6, 12, C] --> [B, 4, 2, 31, 6, 30, 12, C]
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7)
        # reshape back to original shape: [B, 4, 2, 31, 6, 30, 12, C] -> [B, 8, 186, 360, C]
        x = x.reshape(original_shape)
        return x

    def generate_attention_mask(self, zhw=None, device='cpu', fill_value=-100.):
        # Prepare the attention mask, based on 2D Swin-Transformer but adjusted for 3D sphere:
        # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L223
        # Note: pangu-pytorch computes the slices incorrectly, to reproduce their exact results, set reproduce_mask=True

        # We shift by half the window size -> the pixels in all windows expect for last in each dimension (Z & H) are
        # still adjacent. Note that W corresponds to the longitude and due to the earth being a sphere, all pixels in
        # all windows are still connected after the shift and thus do not need to be masked.
        def get_slices(window_size, shift):
            # 3 slices: 1. all but last window, 2. first half of last window, 3. second half of last window
            return [slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None)]

        z_slices = get_slices(self.window_size[0], self.window_size[0] // 2)
        h_slices = get_slices(self.window_size[1], self.window_size[1] // 2)
        if self.reproduce_mask:
            h_slices[1] = slice(self.window_size[1], -self.window_size[1] // 2)

        if zhw is None:
            zhw = [dim + (-dim % size) for dim, size in zip(self.zhw, self.window_size)]
        mask = torch.zeros(*zhw, device=device)

        cnt = 0
        for z_slice in z_slices:
            for h_slice in h_slices:
                mask[z_slice, h_slice, :] = cnt
                cnt += 1

        # partition mask into windows: [1, z, h, w, 1] -> [w // window_size[2], type_of_windows, window_dim]
        mask_windows = self.window_partition(mask.view(1, *zhw, 1)).squeeze()
        # for each window: window_dim x window_dim adjacency mask (0 where adjacent)
        attention_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
        # fill all non-zeros with a negative fill_value (default -100)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, fill_value)
        return attention_mask

    def forward(self, x):
        # Example shapes for x = [B, 521280, C] with Z, H, W = 8, 181, 360 and window_size = [2, 6, 12]
        # Input shape [B, Z * H * W, C], save the shortcut for skip-connection
        shortcut = x

        # Reshape input to three dimensions: [B, Z * H * W, C] -> [B, Z, H, W, C] = [B, 8, 181, 360, C]
        x = x.view(x.shape[0], *self.zhw, x.shape[2])
        # Pad to window size: [B, 8, 181, 360, 192] -> [B, 8, 186, 360, 192]
        x = pad_to_shape(x, (1, *self.window_size, 1))
        original_shape = x.shape  # remember shape for later [B, 8, 186, 360, 192]

        # 3D SwinTransformer: shift windows every other block (set via self.roll in __init__) to connect patches in
        # between different windows, in contrast to the original SwinTransformer, Pangu uses 3D windows
        if self.roll:
            # Shift by half a window in all 3 dimensions Z, H, W
            x = torch.roll(x, shifts=[-size // 2 for size in self.window_size], dims=(1, 2, 3))
            # mask out non-adjacent pixels
            mask = self.generate_attention_mask(x.shape[1:4], x.device)
        else:  # if not shifting, no mask needed
            mask = None

        # Reshape to windows: [B, 8, 186, 360, 192] -> [B, 30, 124, 144, 192]
        x = self.window_partition(x)
        # Apply 3D window attention with earth-specific bias
        x = self.attention(x, mask)

        # Revert back from windows: [B, 30, 124, 144, 192] -> [B, 8, 186, 360, 192]
        x = self.window_reverse(x, original_shape)

        # Revert shifted windows by shifting in the other direction
        if self.roll:
            x = torch.roll(x, shifts=[size // 2 for size in self.window_size], dims=(1, 2, 3))

        # Crop to revert zero-padding [B, 8, 186, 360, 192] -> [B, 8, 181, 360, 192] = [B, Z, H, W, C]
        z, h, w = self.zhw
        x = x[:, :z, :h, :w, :]

        # Reshape back to input shape [B, Z, H, W, C] -> [B, Z * H * W, C]
        x = x.reshape(shortcut.shape)

        # Main calculation stages
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.linear(x)))

        return x


class EarthAttention3D(torch.nn.Module):
    """
    3D window attention with Earth-Specific bias, based on Swin-Transformer's 2D window attention, extended to 3D and
    replacing the relative position bias with an earth-specific bias.
    """
    def __init__(self, dim, num_heads, dropout_rate, window_size, type_of_windows):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_dim = int(np.prod(self.window_size))
        self.type_of_windows = type_of_windows  # the number of different windows
        self.scale = (dim // num_heads) ** -0.5

        self.linear1 = torch.nn.Linear(dim, dim * 3)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Create earth specific bias as trainable parameters and initialize with truncated normal distribution
        # Note that the size of the earth-specific bias differs from that given in the pseudocode. Instead, we use the
        # same size as pangu-pytorch which matches the pretrained weights. This also means we don't construct or use
        # the position index (has already been applied to the pretrained weights?)
        self.earth_specific_bias = torch.zeros(
            1, self.type_of_windows, self.num_heads, self.window_dim, self.window_dim)
        self.earth_specific_bias = torch.nn.Parameter(self.earth_specific_bias)
        timm.layers.trunc_normal_(self.earth_specific_bias, std=0.02)

    def forward(self, x, mask):
        # example shapes given for dim = 192 and num_heads = 6
        # Input x of shape [B, 30, 124, 144, 192], mask of shape [30, 124, 144, 144] (or None)
        original_shape = x.shape  # record original shape for later

        # Create query, key and value
        x = self.linear1(x)  # [B, 30, 124, 144, 192] -> [B, 30, 124, 144, 576]
        # Reshape into query, key and value [B, 30, 124, 144, 576] -> [B, 30, 124, 144, 3, 6, 32]
        qkv = x.view(x.shape[:4] + (3, self.num_heads, self.dim // self.num_heads))
        # Move qkv dimension to front and split into query, key, and value
        # [B, 30, 124, 144, 3, 6, 32] -> 3 x [B, 30, 124, 6, 144, 32]
        query, key, value = qkv.permute((4, 0, 1, 2, 5, 3, 6))

        # Scale and compute attention
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)  # -> [B, 30, 124, 6, 144, 144]

        # Add learnable earth-specific bias to the attention matrix to fix the nonuniformity of the grid
        # Note: for now, we follow the pangu-pytorch implementation and don't reindex the earth-specific bias
        # with the positional index
        attention = attention + self.earth_specific_bias

        # Apply masked attention: mask attention between non-adjacent pixels by adding -100 to the masked element
        if mask is not None:
            # reshape mask: [30, 124, 144, 144] -> [1, 30, 124, 1, 144, 144]
            attention = attention + mask.unsqueeze(2).unsqueeze(0)

        attention = self.softmax(attention)  # [B, 30, 124, 6, 144, 144]
        attention = self.dropout(attention)  # [B, 30, 124, 6, 144, 144]

        # Calculated the tensor after spatial mixing.
        # TODO: transpose value or not? pseudocode vs pangu-pytorch -> for now following pangu-pytorch
        x = attention @ value  # -> [B, 30, 124, 6, 144, 32]

        # Reshape to original shape
        x = torch.permute(x, (0, 1, 2, 4, 3, 5))  # [B, 30, 124, 6, 144, 32] -> [B, 30, 124, 144, 6, 32]
        x = x.reshape(original_shape)  # [B, 30, 124, 144, 6, 32] -> [B, 30, 124, 144, 192]

        # Apply second linear and dropout: [B, 30, 124, 144, 192] -> [B, 30, 124, 144, 192]
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim * 4)
        self.linear2 = torch.nn.Linear(dim * 4, dim)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
