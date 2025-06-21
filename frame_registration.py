'''
Description: 
Author: zxy
Date: 2023-07-14 09:48:00
LastEditors: zxy
LastEditTime: 2024-01-02 11:36:57
'''
import pcammls
from pcammls import *
import cv2
import numpy
import sys
import os


class PythonPercipioDeviceEvent(pcammls.DeviceEvent):
    Offline = False

    def __init__(self):
        pcammls.DeviceEvent.__init__(self)

    def run(self, handle, eventID):
        if eventID == TY_EVENT_DEVICE_OFFLINE:
            print('=== Event Callback: Device Offline!')
            self.Offline = True
        return 0

    def IsOffline(self):
        return self.Offline


def main():
    cl = PercipioSDK()

    dev_list = cl.ListDevice()
    for idx in range(len(dev_list)):
        dev = dev_list[idx]
        print('{} -- {} \t {}'.format(idx, dev.id, dev.iface.id))
    if len(dev_list) == 0:
        print('no device')
        return
    if len(dev_list) == 1:
        selected_idx = 0
    else:
        selected_idx = int(input('select a device:'))
    if selected_idx < 0 or selected_idx >= len(dev_list):
        return

    sn = dev_list[selected_idx].id

    handle = cl.Open(sn)
    if not cl.isValidHandle(handle):
        err = cl.TYGetLastErrorCodedescription()
        print('no device found : ', end='')
        print(err)
        return

    event = PythonPercipioDeviceEvent()
    cl.DeviceRegiststerCallBackEvent(event)

    color_fmt_list = cl.DeviceStreamFormatDump(handle, PERCIPIO_STREAM_COLOR)
    if len(color_fmt_list) == 0:
        print('device has no color stream.')
        return

    print('color image format list:')
    for idx in range(len(color_fmt_list)):
        fmt = color_fmt_list[idx]
        print('\t{} -size[{}x{}]\t-\t desc:{}'.format(idx, cl.Width(fmt), cl.Height(fmt), fmt.getDesc()))
    cl.DeviceStreamFormatConfig(handle, PERCIPIO_STREAM_COLOR, color_fmt_list[0])

    depth_fmt_list = cl.DeviceStreamFormatDump(handle, PERCIPIO_STREAM_DEPTH)
    if len(depth_fmt_list) == 0:
        print('device has no depth stream.')
        return

    print('depth image format list:')
    for idx in range(len(depth_fmt_list)):
        fmt = depth_fmt_list[idx]
        print('\t{} -size[{}x{}]\t-\t desc:{}'.format(idx, cl.Width(fmt), cl.Height(fmt), fmt.getDesc()))
    cl.DeviceStreamFormatConfig(handle, PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])

    err = cl.DeviceLoadDefaultParameters(handle)
    if err:
        print('Load default parameters fail: ', end='')
        print(cl.TYGetLastErrorCodedescription())
    else:
        print('Load default parameters successful')

    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
    print('depth image scale unit :{}'.format(scale_unit))
    # 读取校准数据
    depth_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    color_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_COLOR)

    err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
    if err:
        print('device stream enable err:{}'.format(err))
        return

    print('{} -- {} \t'.format(0, "Map depth to color coordinate(suggest)"))
    print('{} -- {} \t'.format(1, "Map color to depth coordinate"))
    # 模式0：将深度图映射到彩色图坐标系（常用，彩色图分辨率通常更高）。
    # 模式1：将彩色图映射到深度图坐标系。
    registration_mode = int(input('select registration mode(0 or 1):'))
    if selected_idx < 0 or selected_idx >= 2:
        registration_mode = 0

    cl.DeviceStreamOn(handle)
    img_registration_depth = image_data()
    img_registration_render = image_data()
    img_parsed_color = image_data()
    img_undistortion_color = image_data()
    img_registration_color = image_data()
    while True:
        if event.IsOffline():
            break
        image_list = cl.DeviceStreamRead(handle, 2000)
        if len(image_list) == 2:
            for i in range(len(image_list)):
                frame = image_list[i]
                if frame.streamID == PERCIPIO_STREAM_DEPTH:
                    img_depth = frame
                if frame.streamID == PERCIPIO_STREAM_COLOR:
                    img_color = frame

            if 0 == registration_mode:  # 深度图映射到彩色图坐标系
                cl.DeviceStreamMapDepthImageToColorCoordinate(depth_calib, img_depth, scale_unit, color_calib,
                                                              img_color.width, img_color.height, img_registration_depth)

                cl.DeviceStreamDepthRender(img_registration_depth, img_registration_render)

                # 深度映射到彩色坐标系：利用校准数据将深度像素对齐到彩色图像。
                # 渲染深度图：将16位深度数据转为8位伪彩色图，便于显示。
                mat_depth_render = img_registration_render.as_nparray()
                cv2.imshow('registration', mat_depth_render)

                cl.DeviceStreamImageDecode(img_color, img_parsed_color)
                cl.DeviceStreamDoUndistortion(color_calib, img_parsed_color, img_undistortion_color)
                mat_undistortion_color = img_undistortion_color.as_nparray()
                cv2.imshow('undistortion rgb', mat_undistortion_color)
            else:  # 彩色图映射到深度图坐标系。
                cl.DeviceStreamImageDecode(img_color, img_parsed_color)
                cl.DeviceStreamDoUndistortion(color_calib, img_parsed_color, img_undistortion_color)

                # 将彩色图像映射到深度坐标系，适用于需要深度图分辨率的场景。
                cl.DeviceStreamMapRGBImageToDepthCoordinate(depth_calib, img_depth, scale_unit, color_calib,
                                                            img_undistortion_color, img_registration_color)

                cl.DeviceStreamDepthRender(img_depth, img_registration_render)
                mat_depth_render = img_registration_render.as_nparray()
                cv2.imshow('depth', mat_depth_render)

                mat_registration_color = img_registration_color.as_nparray()
                cv2.imshow('registration rgb', mat_registration_color)

            k = cv2.waitKey(10)
            if k == ord('q'):
                break

    cl.DeviceStreamOff(handle)
    cl.Close(handle)
    pass


if __name__ == '__main__':
    main()
