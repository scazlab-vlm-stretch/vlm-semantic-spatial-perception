import math
import rospy
from sensor_msgs.msg import JointState

class Record:
    """
    This class is used to record the "significant" waypoints of a robot's joints during its movement. It stores the joint names and their corresponding waypoints, which include position, velocity, and timestamp information. 
    The class also has flags to indicate whether recording has started or stopped.

    joint_name: [waypoint1, waypoint2, ...]
    waypoint: [position, velocity, time_stamp, header_stamp]

    header_stamp is the time the joint is idle in between waypoints.

    Note-to-self: the waypoints are recorded when the direction of the joint movement changes. However, the specific velocity/acceleration with which it moves is not recorded (because the velocity is 0 at these points)
    To add such functionality, perhaps record the velocity + acceleration curve throughout the motion trajectory, and record velocity reaches a maximum, and create another wayponit there with that specific velocity.
    """
    
    def __init__(self):
        self.start = False
        self.stop = False

        self.start_time = 0

        # self.velocity_threshold = 0.05  # Threshold for detecting significant velocity changes
        # note: in position mode, Stretch position commands are tracked by a trapezoidal motion profile.

        self.waypoints = {} 
        self.pauses = {}

    def start_recording(self, time):
        """
        Start recording the waypoints.
        """
        self.start = True
        self.stop = False

        self.start_time = time

    def stop_recording(self):
        """
        Stop recording the waypoints.
        """
        self.stop = True

    def detect_waypoint(self, joint_name, position, velocity, time_stamp):
        if self.start:
            if joint_name in self.waypoints:
                prev_waypoint = self.waypoints[joint_name][-1]
                prev_joint_direction = math.copysign(1, prev_waypoint[0]) \
                    if len(self.waypoints[joint_name]) < 2 \
                        else math.copysign(1, prev_waypoint[0] - self.waypoints[joint_name][-2][0])  # Get the direction of the previous joint velocity (don't use velocity itself because it jumps between 0.001 and -0.001 at rest)
                
                if abs(velocity) <= 0.001 and abs(prev_waypoint[1]) > 0.001:
                    self.pauses[joint_name] = time_stamp

                if math.copysign(1, velocity) != prev_joint_direction:
                    header_stamp = time_stamp if joint_name not in self.pauses else time_stamp - self.pauses[joint_name]
                    self.waypoints[joint_name].append([round(position,3), round(velocity,3), time_stamp, header_stamp])
                    self.pauses.pop(joint_name, None)  # Remove the pause entry for this joint
                    
            else:
                self.waypoints[joint_name] = []
                self.waypoints[joint_name].append([round(position,3), round(velocity,3), time_stamp, 0])
        elif self.stop and self.start: #stopped recording, but need to record the last waypoint
            self.start = False
            if joint_name in self.waypoints:
                header_stamp = time_stamp if joint_name not in self.pauses else time_stamp - self.pauses[joint_name]
                self.waypoints[joint_name].append([round(position,3), round(velocity,3), time_stamp, header_stamp])
                self.pauses.pop(joint_name, None)  # Remove the pause entry for this joint

            self.print_waypoints()
            

    def get_waypoints(self):
        """
        Get the recorded waypoints.
        """
        return self.waypoints
    
    def clear_waypoints(self):
        """
        Clear the recorded waypoints.
        """
        self.waypoints = {}
    
    def print_waypoints(self):
        """
        Print the recorded waypoints in JSON format.
        """
        import json
        print(json.dumps(self.waypoints, indent=4))

    def save_waypoints(self, file_path):
        """
        Save the recorded waypoints to a JSON file.
        """
        import json
        with open(file_path, 'w') as f:
            json.dump(self.waypoints, f, indent=4)

    def callback(self, data):
        """
        Callback function to process incoming JointState messages.
        """
        for index, name in enumerate(data.name):
            joint_name = data.name[index]
            position = data.position[index]
            velocity = data.velocity[index]
            time_stamp = rospy.Time.now() - self.start_time
            self.detect_waypoint(joint_name, position, velocity, time_stamp.to_sec())

    def get_time_elapsed(self):
        return rospy.Time.now() - self.start_time

    def is_recording(self):
        return self.start == True

    def main(self):
        """
        Main function to run the recording process.
        """
        rospy.init_node('record_waypoints', anonymous=True)
        rospy.loginfo("Recording waypoints...")

        rospy.Subscriber('/stretch/joint_states', JointState, self.callback)

if __name__ == "__main__":
    record = Record()
    record.main()

    record.start_recording(rospy.Time.now())
    
    rate = rospy.Rate(10)

    try:
        print("Is recording?", record.is_recording(), flush=True)
        while not rospy.is_shutdown() and record.get_time_elapsed().to_sec() < 4:
            rate.sleep()

        record.stop_recording()

        print("Is recording?", record.is_recording(), flush=True)
        record.print_waypoints()
        record.save_waypoints("/home/fern-stretch/vlm-semantic-spatial-perception/outputs/stretch_record/waypoints.json")

    except KeyboardInterrupt:
        print("WAYPOINTS:\n", flush=True)
        record.print_waypoints()

    