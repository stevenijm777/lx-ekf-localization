#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import tf
from multiprocessing import Lock
from typing import Optional

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.camera.types import Rectifier
from dt_apriltags import Detector
from turbojpeg import TurboJPEG
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance
from solution.ekf import EKF
from ekf_localization.include.odometry_utils import delta_phi, get_odometry
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown.utils.image.ros import compressed_imgmsg_to_rgb, rgb_to_compressed_imgmsg


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi


class EKFLocalizationNode(DTROS):
    """
    Publisher: ~/estimate_pose (:obj:`PoseWithCovarianceStamped`)

    Subscribers:
        ~/image/compressed (:obj:`CompressedImage`):
            compressed image
    """
    right_tick_prev: Optional[int]
    left_tick_prev: Optional[int]
    delta_phi_left: float
    delta_phi_right: float


    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(EKFLocalizationNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.loginfo("Initializing...")
        # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")
        self.no_predict = rospy.get_param("~no_predict", False)
        self.no_update = rospy.get_param("~no_update", False)
        self.sim = rospy.get_param("~test_sim", False)
        self.right_wheel_mutex = Lock()
        self.left_wheel_mutex = Lock()
        self.gt_pose = None
        self.latest_img = None

        # Init the parameters
        self.resetParameters()

        # nominal R and L, you may change these if needed:

        self.R = 0.0318  # meters, default value of wheel radius
        self.baseline = 0.11  # meters, default value of baseline for DB21
        self.camera_model = None
        self.rectifier = None
        self.rect_camera_K = None
        self.jpeg = TurboJPEG()

        q_0 = np.array([
            rospy.get_param("~x_0", 0.0),
            rospy.get_param("~y_0", 0.0),
            rospy.get_param("~theta_0", 0.0),
        ])

        P_0 = np.array([
            [ rospy.get_param("~P_0_xx", 0.0), 0.0, 0.0 ],
            [ 0.0, rospy.get_param("~P_0_yy", 0.0), 0.0 ],
            [ 0.0, 0.0, rospy.get_param("~P_0_tt", 0.0) ],
        ])


        Q = np.array([
            [ rospy.get_param("~Q_xx", 0.0), 0.0],
            [ 0.0, rospy.get_param("~Q_tt", 0.0) ],
        ])

        R = np.array([
            [ rospy.get_param("~R_rr", 0.0), 0.0 ],
            [ 0.0, rospy.get_param("~R_tt", 0.0) ],
        ])
        self.ekf = EKF(q_0, P_0, Q, R)

        map_file = rospy.get_param("~map", None)
        if map_file is None:
            rospy.logerr("No map provided")
            rospy.signal_shutdown("No map provided")

        self.map = {int(id): np.array(value["position"]) for id, value   in map_file.items()}

        self.apriltag_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25
        )

        # Defining subscribers:
        rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            buff_size=10000000,
            queue_size=1,
        )

        self.sub_camera_info = rospy.Subscriber(
            f"/{self.veh}/camera_node/camera_info",
            CameraInfo,
            self.cb_info,
            queue_size=1,
        )

        # Wheel encoder subscriber:
        left_encoder_topic = f"/{self.veh}/left_wheel_encoder_driver_node/tick"
        rospy.Subscriber(left_encoder_topic,
                         WheelEncoderStamped,
                         self.cbLeftEncoder)

        right_encoder_topic = f"/{self.veh}/right_wheel_encoder_driver_node/tick"
        rospy.Subscriber(right_encoder_topic,
                         WheelEncoderStamped,
                         self.cbRightEncoder)

        self.sub_gt_pose = rospy.Subscriber(
            f"/{self.veh}/duckiematrix_interface_node/state",
            Odometry,
            self.cbGTPose,
            queue_size=1,
        )

        self.pub_detections = rospy.Publisher(
            f"/{self.veh}/detections/image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
            dt_help="Camera image with tag publishes superimposed",
            latch=True
        )

        self.pub_pose_covariance = rospy.Publisher(
            f"/{self.veh}/ekf_localization_node/pose",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
            latch=True
        )

        self.pub_landmark_markers = rospy.Publisher(
            f"/{self.veh}/map_markers",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        # Need to sleep for a bit for the publisher to register with master
        rospy.sleep(0.5)
        self.publish_landmarks([])
        self.publish_pose()

        # we will do prediction at a fixed frequency rather than asynchronously
        # when the encoder data arrives
        rospy.Timer(rospy.Duration(1.0/10.0), self.doPredict)


        self.loginfo("Initialized!")

    def cbGTPose(self, odom_msg):
        q = odom_msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        p = odom_msg.pose.pose.position
        self.gt_pose = [p.x, p.y, yaw]

    def cbLeftEncoder(self, encoder_msg):
        """
        Wheel encoder callback
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """
        with self.left_wheel_mutex:
            # initializing ticks to stored absolute value
            if self.left_tick_prev is None:
                self.left_tick_prev = encoder_msg.data
                return

            left_ticks_curr = encoder_msg.data
            # running the DeltaPhi() function copied from the notebooks to calculate rotations
            delta_phi_left = delta_phi(
                left_ticks_curr, self.left_tick_prev, encoder_msg.resolution
            )
            if delta_phi_left == 0:
                return
            self.left_tick_prev = left_ticks_curr
            self.delta_phi_left += delta_phi_left

    def cbRightEncoder(self, encoder_msg):
        """
        Wheel encoder callback, the rotation of the wheel.
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """

        with self.right_wheel_mutex:
            if self.right_tick_prev is None:
                self.right_tick_prev = encoder_msg.data
                return

            right_ticks_curr = encoder_msg.data

            # calculate rotation of right wheel
            delta_phi_right = delta_phi(
                right_ticks_curr, self.right_tick_prev, encoder_msg.resolution
            )
            if delta_phi_right == 0:
                return
            self.right_tick_prev = right_ticks_curr
            self.delta_phi_right += delta_phi_right


    def doPredict(self, event=None):


        if self.delta_phi_right == 0 and self.delta_phi_left ==0:
            # we haven't moved no need to predict
            return

        with self.left_wheel_mutex:
            with self.right_wheel_mutex:


                dX, dT = get_odometry(
                    self.R,
                    self.baseline,
                    self.delta_phi_left,
                    self.delta_phi_right
                )

                self.ekf.predict(dX, dT)
                self.delta_phi_left = 0
                self.delta_phi_right = 0
                self.doUpdate()


    def cb_info(self, msg):
        self.loginfo("Camera info message received. Unsubscribing from camera_info topic.")
        try:
            self.sub_camera_info.shutdown()
        except BaseException:
            pass
        H, W = msg.height, msg.width
        # create new camera info
        self.camera_model = CameraModel(
            width=W,
            height=H,
            K=np.reshape(msg.K, (3, 3)),
            D=np.reshape(msg.D, (5,)),
            P=np.reshape(msg.P, (3, 4)),
        )
        self.rectifier = Rectifier(self.camera_model)
        self.rect_camera_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_model.K, self.camera_model.D, (W, H), 0.0
        )



    def cb_image(self, image_msg):
        self.latest_img = image_msg

    def doUpdate(self):

        """

        Args:
            image_msg (:obj:`sensor_msgs.msg.CompressedImage`): The received image message

        """
        if self.no_update:
            return

        if self.camera_model is None:
            return

        if self.latest_img is None:
            print("waiting for first image")
            return
        
        # Decompress the image
        image_rgb = compressed_imgmsg_to_rgb(self.latest_img)

        rect_image = self.rectifier.rectify(image_rgb, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale for AprilTag detection
        #rect_image_gray = cv2.cvtColor(rect_image, cv2.COLOR_RGB2GRAY)
        image_gray = cv2.cvtColor(rect_image, cv2.COLOR_RGB2GRAY)

        # Camera parameters for pose estimation
        fx = self.camera_model.K[0, 0]
        fy = self.camera_model.K[1, 1]
        cx = self.camera_model.K[0, 2]
        cy = self.camera_model.K[1, 2]
        camera_params = [fx, fy, cx, cy]
        
        # AprilTag size in meters (Duckietown standard)
        tag_size = 0.065
        
        # Detect AprilTags with pose estimation
        detections = self.apriltag_detector.detect(
            image_gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size
        )
        
        # Process each detection
        for detection in detections:
            tag_id = detection.tag_id
            
            # Check if this tag is in our map
            if tag_id not in self.map:
                continue
            
            # Get the tag position from the map
            tag_position = self.map[tag_id]
            tag_x, tag_y = tag_position[0], tag_position[1]

            # let's calculate the exact range and bearing using the GT pose and
            # tag location
            if self.gt_pose is None:
                return
            dx = tag_x - self.gt_pose[0]
            dy = tag_y - self.gt_pose[1]
            sim_range_estimate = np.linalg.norm([dx, dy])
            sim_bearing = np.arctan2(dy, dx) - self.gt_pose[2]
            sim_bearing = wrap_angle(sim_bearing)

            # Get pose from detection (translation vector in camera frame)
            # pose_t is a 3x1 matrix: [x, y, z] where z is forward, x is right, y is down
            t = detection.pose_t

            # Extract range and bearing from the pose
            # Range is the distance to the tag
            range_estimate = np.linalg.norm(t)

            # Bearing is the angle in the horizontal plane (around y-axis)
            # arctan2(x, z) gives the angle from camera's forward direction
            bearing = -np.arctan2(t[0, 0], t[2, 0])
            bearing=wrap_angle(bearing)

            if self.sim:
                range_estimate = sim_range_estimate
                bearing = sim_bearing

            # Update the EKF with this measurement
            self.ekf.update([range_estimate, bearing], [tag_x, tag_y])
        ids = [det.tag_id for det in detections]
        self.publish_landmarks(ids)
        self.publish_detections(image_gray, detections, self.latest_img.header)
        self.publish_pose(self.latest_img.header)

    def publish_pose(self, header=None):

        pose_cov= PoseWithCovariance()
        pose_cov.pose.position.x = self.ekf.q[0]
        pose_cov.pose.position.y = self.ekf.q[1]
        pose_cov.pose.position.z = 0.0

        pose_cov.pose.orientation.x = 0.0
        pose_cov.pose.orientation.y = 0.0
        pose_cov.pose.orientation.z = np.sin(self.ekf.q[2] / 2)
        pose_cov.pose.orientation.w = np.cos(self.ekf.q[2] / 2)

        pose_cov.covariance = [
            self.ekf.P[0, 0], self.ekf.P[0, 1], 0.0, 0.0, 0.0, self.ekf.P[0, 2] ,
            self.ekf.P[1, 0], self.ekf.P[1, 1], 0.0, 0.0, 0.0, self.ekf.P[1, 2] ,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            self.ekf.P[2, 0], self.ekf.P[2, 1], 0.0, 0.0, 0.0, self.ekf.P[2, 2]
        ]
        odom_msg = Odometry()
        if header is None:
            odom_msg.header.stamp = rospy.Time.now()
        else:
            odom_msg.header = header
        odom_msg.header.frame_id = "map"
        odom_msg.pose = pose_cov

        self.pub_pose_covariance.publish(odom_msg)


    def resetParameters(self):

        self.log("Encoder data resetting")
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

    def publish_landmarks(self, detection_ids):
        # publishes the whole map as markers and colors the ones that were detected

        marker_array = MarkerArray()
        for i, (landmark_id, position) in enumerate(self.map.items()):


            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = "landmarks"
            m.id = int(landmark_id)
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = position[0]
            m.pose.position.y = position[1]
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0

            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            if landmark_id in detection_ids:
                m.color.r = 0.0
                m.color.g = 0.0
                m.color.b = 1.0
                m.color.a = 1.0
            else:
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 1.0


            marker_array.markers.append(m)
        self.pub_landmark_markers.publish(marker_array)

    def publish_detections(self, img, detections, header):

        # get a color buffer from the BW image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # draw each tag
        for detection in detections:
            for idx in range(len(detection.corners)):
                cv2.line(
                    img,
                    tuple(detection.corners[idx - 1, :].astype(int)),
                    tuple(detection.corners[idx, :].astype(int)),
                    (0, 255, 0),
                )
            # draw the tag ID
            cv2.putText(
                img,
                str(detection.tag_id),
                org=(detection.corners[0, 0].astype(int) + 10, detection.corners[0, 1].astype(int) + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
            )
        # pack image into a message
        img_msg = CompressedImage()
        img_msg.header.stamp = header.stamp
        img_msg.header.frame_id = header.frame_id
        img_msg.format = "jpeg"
        img_msg.data = self.jpeg.encode(img)
        # ---
        self.pub_detections.publish(img_msg)


if __name__ == "__main__":
    # Initialize the node
    encoder_localization_node = EKFLocalizationNode(node_name="ekf_localization_node")
    # Keep it spinning
    rospy.spin()


