#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import json
from geometry_msgs.msg import Twist
from websocket import create_connection

def websocket_to_cmd_vel(ws_url):
    """
    Łączy się z serwerem WebSocket i przesyła dane do topicu cmd_vel w ROS.
    
    :param ws_url: URL serwera WebSocket (np. ws://localhost:9000)
    """
    rospy.init_node('websocket_cmd_vel_node', anonymous=True)
    cmd_vel_publisher = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=10)
    
    try:
        # Połączenie z serwerem WebSocket
        ws = create_connection(ws_url)
        rospy.loginfo("Połączono z serwerem WebSocket: {}".format(ws_url))
        rospy.loginfo(rospy.is_shutdown())
        
        while not rospy.is_shutdown():
            # Odbieranie wiadomości z WebSocket
            rospy.loginfo("Waiting for message: {}".format(ws_url))
            message = ws.recv()
            rospy.loginfo("Odebrano wiadomość: {}".format(message))
            
            try:
                # Parsowanie wiadomości jako JSON
                data = json.loads(message)
                twist = Twist()
                twist.linear.x = data.get('x', 0.0)
                twist.linear.y = data.get('y', 0.0)
                twist.linear.z = data.get('z', 0.0)
                
                # Publikowanie danych do topicu cmd_vel
                cmd_vel_publisher.publish(twist)
                rospy.loginfo("Opublikowano Twist: {}".format(twist))
            except Exception as e:
                rospy.logerr("Błąd podczas parsowania wiadomości: {}".format(e))
    
    except rospy.ROSInterruptException:
        rospy.loginfo("Node został zatrzymany.")
    except Exception as e:
        rospy.logerr("Błąd połączenia z WebSocket: {}".format(e))
    finally:
        ws.close()
        rospy.loginfo("Połączenie z WebSocket zamknięte.")

if __name__ == "__main__":
    try:
        websocket_url = "ws://host.docker.internal:8080"  # Zmień na odpowiedni URL serwera WebSocket
        websocket_to_cmd_vel(websocket_url)
    except rospy.ROSInterruptException:
        pass
