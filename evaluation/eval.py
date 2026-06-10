#!/usr/bin/env python3
import os
import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

class RGBDepthSyncRecorder:
    def __init__(self):
        self.bridge = CvBridge()

        # ===== ROS params =====
        self.rgb_topic   = rospy.get_param("~rgb_topic",   "/RGB")
        self.depth_topic = rospy.get_param("~depth_topic", "/depth")
        self.save_dir    = rospy.get_param("~save_dir",    "/tmp/rgb_depth_data")

        self.rgb_dir   = os.path.join(self.save_dir, "rgb")
        self.depth_dir = os.path.join(self.save_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # ===== Subscribers with time sync =====
        rgb_sub   = Subscriber(self.rgb_topic, Image)
        depth_sub = Subscriber(self.depth_topic, Image)

        self.sync = ApproximateTimeSynchronizer(
            fs=[rgb_sub, depth_sub],
            queue_size=20,
            slop=0.03,          # 允许 30ms 误差，可按相机帧率调
            allow_headerless=False
        )
        self.sync.registerCallback(self.sync_callback)

        rospy.loginfo("RGB-Depth synchronized recorder started")
        rospy.loginfo(f"RGB topic   : {self.rgb_topic}")
        rospy.loginfo(f"Depth topic : {self.depth_topic}")
        rospy.loginfo(f"Save dir    : {self.save_dir}")

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg, desired_encoding="bgr8"
            )
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        # 使用 RGB 的时间戳作为该 frame 的时间
        stamp = rgb_msg.header.stamp.to_sec()
        filename = f"{stamp:.6f}.png"

        rgb_path   = os.path.join(self.rgb_dir, filename)
        depth_path = os.path.join(self.depth_dir, filename)

        cv2.imwrite(rgb_path, rgb)
        cv2.imwrite(depth_path, depth)

        rospy.loginfo_throttle(
            1.0,
            f"Saved synchronized frame @ {stamp:.6f}"
        )

if __name__ == "__main__":
    rospy.init_node("rgb_depth_sync_recorder")
    RGBDepthSyncRecorder()
    rospy.spin()
