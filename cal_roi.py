import time

import pcammls
from pcammls import *
import cv2
import numpy

def get_roi():
    """
    从Percipio设备获取彩色图像，然后框选ROI区域并打印左上、右下坐标点信息供detect.py使用
    使用方法：
        运行此脚本后，将弹出一个窗口显示彩色图像，按下'q'键退出，按下'c'键保存当前图像并进入ROI选择模式。
        框选ROI区域后回车，脚本将打印出ROI区域的左上和右下坐标点，并返回这些坐标点列表。
    返回:
        List[Tuple[int, int, int, int]]: 返回ROI区域的左上和右下坐标点列表
    作者:
        段宇辉
    最后修改时间:
        2025-06-21
    """

    select = input("输入1框选彩色图像ROI区域，输入2框选深度图像ROI区域，其他键退出: ")
    # 1. 初始化SDK
    cl = PercipioSDK()

    # 2. 查找并打开设备
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

    # 准备用于接收图像数据的对象
    rgb_image = image_data()
    depth_render = image_data()

    # 5. 循环读取和处理图像
    while True:
        image_list = cl.DeviceStreamRead(handle, -1)
        if image_list:
            for frame in image_list:

                if frame.streamID == PERCIPIO_STREAM_COLOR:
                    # 处理彩色图
                    cl.DeviceStreamImageDecode(frame, rgb_image)
                    color_arr = rgb_image.as_nparray()
                    cv2.imshow('color', color_arr)

                if frame.streamID == PERCIPIO_STREAM_DEPTH:
                    # 处理深度图
                    cl.DeviceStreamDepthRender(frame, depth_render)
                    depth_arr = depth_render.as_nparray()
                    cv2.imshow('depth', depth_arr)

        key = cv2.waitKey(100)
        if key == ord('q'):
            break
        elif key == ord('c'):
            if select == '1':
                print("已保存彩色图像，准备框选ROI区域")
                cv2.imwrite('color_image_roi.png', color_arr)
            elif select == '2':
                print("已保存深度图像，准备框选ROI区域")
                cv2.imwrite('color_image_roi.png', depth_arr)
            # 保存图片

            break

    # 6. 停止数据流并关闭设备
    cl.DeviceStreamOff(handle)
    cl.Close(handle)
    cv2.destroyAllWindows()
    # 7. 框选ROI区域
    # 返回的roi格式为(x, y, w, h)，其中(x, y)是左上角坐标，w是宽度，h是高度
    if select == '1':
        roi = cv2.selectROI("Select ROI", color_arr, fromCenter=False, showCrosshair=True)
    elif select == '2':
        roi = cv2.selectROI("Select ROI", depth_arr, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        print("未选择有效的ROI区域")
        return []

    #删除临时生成的图片
    import os
    if os.path.exists('color_image_roi.png'):
        os.remove('color_image_roi.png')
    return [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]

if __name__ == '__main__':
    roi = get_roi()
    print(f"ROI区域左上角x，y: ({roi[0]}, {roi[1]}), 右下角x，y: ({roi[2]}, {roi[3]})")
    # 示例输出: ROI区域左上角: (100, 150), 右下角: (400, 450)