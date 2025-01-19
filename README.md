# Bebop drone gesture control project

### Wymagane oprogramowanie:

Opracowany przez nas program wymaga ros-melodic oraz paczek wymienionych poniżej.

### Wymagane paczki:
- [symulator bebop](https://github.com/simonernst/iROS_drone/tree/melodic)
- [paczki zalecane przez producenta](https://bebop-autonomy.readthedocs.io/en/latest/installation.html)

Zalecamy instalację paczek wskazanych przez producenta, a następnie dodanie paczki symulatora, bez joystick_drivers.

### Procedura uruchomienia programu:

1) Uruchom symulator poleceniem `roslaunch rotors_gazebo mav_velocity_control_with_fake_driver.launch`,
2) uruchom skrypt gestures_recognition.py. Jest to zwyczajny skrypt, niewymagający ROSa. Ze względu na brak kompatybilności, nie powinien on być uruchamiany w ros-melodic,
3) Uruchom węzeł odbierający polecenia z websocket'a i przekazujący je do robota za pomocą `rosrun data_receiver data_receiver` 

**Jeśli pracujesz na prawdziwym hardware, użyj komendy `roslaunch bebop_driver bebop_node.launch -- ip:=<DRONE_IP_ADDRESS>` zamiast uruchomienia symulatora**