from pcammls import *


def get_p3d_data(cl):
    dev_list = cl.ListDevice()
    if len(dev_list) == 0:
        print('未找到设备')
        exit()
    handle = cl.Open(dev_list[0].id)
    if not cl.isValidHandle(handle):
        print('打开设备失败')
        exit()

    # 3. 启用彩色和深度数据流
    err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
    if err:
        print('启用数据流失败: {}'.format(err))
        cl.Close(handle)
        exit()

    # 4. 启动数据流
    cl.DeviceStreamOn(handle)

    # 读取校准数据
    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
    print('depth image scale unit :{}'.format(scale_unit))  # 0.125
    depth_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    color_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_COLOR)
    depth_calib_data = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    depth_calib_width = depth_calib_data.Width()
    depth_calib_height = depth_calib_data.Height()
    depth_calib_intr = depth_calib_data.Intrinsic()
    depth_calib_extr = depth_calib_data.Extrinsic()
    depth_calib_dis = depth_calib_data.Distortion()

    # 准备用于接收图像数据的对象
    rgb_image = image_data()
    depth_render = image_data()
    # 点云数据
    pointcloud_data_arr = pointcloud_data_list()