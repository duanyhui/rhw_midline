import pcammls
from pcammls import *
import cv2
import numpy as np
import open3d as o3d


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

    depth_fmt_list = cl.DeviceStreamFormatDump(handle, PERCIPIO_STREAM_DEPTH)
    if len(depth_fmt_list) == 0:
        print('device has no depth stream.')
        return

    cl.DeviceStreamFormatConfig(handle, PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])

    depth_calib_data = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)

    err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_DEPTH)
    if err:
        print('device stream enable err:{}'.format(err))
        return

    cl.DeviceStreamOn(handle)

    # 初始化Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud')
    pcd = o3d.geometry.PointCloud()
    added_geometry = False

    pointcloud_data_arr = pointcloud_data_list()
    while True:
        if event.IsOffline():
            break
        image_list = cl.DeviceStreamRead(handle, -1)

        for frame in image_list:
            if frame.streamID == PERCIPIO_STREAM_DEPTH:
                # 将深度图转换为点云
                cl.DeviceStreamMapDepthImageToPoint3D(frame, depth_calib_data, scale_unit, pointcloud_data_arr)

                # 转换为numpy数组并处理
                p3d_nparray = pointcloud_data_arr.as_nparray()
                height, width, _ = p3d_nparray.shape
                points = p3d_nparray.reshape(-1, 3)

                # 过滤无效点（假设Z<=0为无效）
                valid_mask = points[:, 2] > 0
                valid_points = points[valid_mask]

                if valid_points.size == 0:
                    continue

                # 更新点云数据
                pcd.points = o3d.utility.Vector3dVector(valid_points)

                # 第一次添加几何体，后续更新
                if not added_geometry:
                    vis.add_geometry(pcd)
                    added_geometry = True
                else:
                    vis.update_geometry(pcd)

                # 更新渲染
                vis.poll_events()
                vis.update_renderer()

        # 处理退出按键
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # 关闭窗口和设备
    vis.destroy_window()
    cl.DeviceStreamOff(handle)
    cl.Close(handle)


if __name__ == '__main__':
    main()
