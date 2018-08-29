import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    #i0得出k*k*c个横坐标，例如3*3的kernel，C个channel，则i0为[0,0,0,1,1,1,2,2,2]*channel
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    #一共会出out_height*out_width个像素点，在im2col中为一行，每个像素点是原图的各channel与对应的kernel的卷积的加和
    #i1表示kernel的window纵向滑动量，因为输出每行有out_width个像素点，则每经过out_width个点之后，kernel就进入图的下一行
    #因此out_height=0时，i1=0,到out_height到图最底部时，i1=out_height-1，
    #因此i1形式为[0,....,0,1....,1,............,out_height-1],每个数重复out_width次
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    #j0为kernel window的基础纵坐标，3*3*channel时的形式为[0,1,2,0,1,2,0,1,2]*channel
    j0 = np.tile(np.arange(field_width), field_height * C)
    #窗口纵向滑动量, 形式与i1相反，在out_width这条边行进时滑动量从0->out_width-1，换行时归零
    j1 = stride * np.tile(np.arange(out_width), out_height)
    #reshape(-1,1) -> shape变为n*1的vector,i,j的一列的n个数就是kernel的基础横纵坐标，n=k*k*c
    #reshape(1,-1) -> 变为1*m，依顺序偏移左边, m=out_h*out_w
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    #k为channel的index，每k*k个点进入下一个channel
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    #在C个channel上，各取k*k个点，横纵坐标由i,j控制，共有(k*k*C) * (out_width*out_height)个点
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
