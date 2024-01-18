# Driving Behaviour

## Competition Description

### Context

Aggressive driving behavior is the leading factor of road traffic accidents. As reported by the **AAA Foundation for
Traffic Safety**, 106,727 fatal crashes – 55.7 percent of the total – during a recent four-year period involved drivers
who committed one or more aggressive driving actions. Therefore, how to predict dangerous driving behavior quickly and
accurately?

### Solution Approach

Aggressive driving includes speeding, sudden breaks and sudden left or right turns. All these events are reflected on
accelerometer and gyroscope data. Therefore, knowing that almost everyone owns a smartphone nowadays which has a wide
variety of sensors, we've designed a data collector application in android based on the accelerometer and gyroscope
sensors.

### Content

* Sampling Rate: 2 samples (rows) per second.
* Gravitational acceleration: removed.
* Sensors: Accelerometer and Gyroscope.
* Data:
    * Acceleration (X,Y,Z axis in meters per second squared (m/s2))
    * Rotation (X,Y, Z axis in degrees per second (°/s))
    * Classification label (SLOW, NORMAL, AGGRESSIVE)
    * Timestamp (time in seconds)
* Driving Behaviors:
    * Slow
    * Normal
    * Aggressive
* Device: Samsung Galaxy S21

### Articles

* [Building a Driving Behaviour Dataset](https://rochi.utcluj.ro/articole/10/RoCHI2022-Cojocaru-I-1.pdf)
* [Driver Behaviour Analysis based on Deep Learning Algorithms](https://rochi.utcluj.ro/articole/10/RoCHI2022-Cojocaru-I-2.pdf)

### Authors

* Paul-Stefan Popescu
* Ion Cojocaru

## Solution Approach

Using 2 classifiers:

* Random Forest Classifier (accuracy score of 0.4)
* Logistic Regression (accuracy score of 0.35)
    * R-squared: -1.26  [ -∞ (worse performance) - 0.0 (no fit) - 1.0 (perfect fit) ]

### Output

```shell
C:\Users\emanu\Desktop\Workspace\ai\driving_behaviour\.venv\Scripts\python.exe C:\Users\emanu\Desktop\Workspace\ai\driving_behaviour\main.py 
Classes:  {'SLOW': 0, 'NORMAL': 1, 'AGGRESSIVE': 2}
       Class  ClassCode
0     NORMAL          1
1     NORMAL          1
2     NORMAL          1
3     NORMAL          1
4     NORMAL          1
...      ...        ...
3639    SLOW          0
3640    SLOW          0
3641    SLOW          0
3642    SLOW          0
3643    SLOW          0

[3644 rows x 2 columns]
Model Accuracy (Random Forest): 0.40
Model Accuracy (Logistic Regression): 0.35
R-squared (Logistic Regression): -1.26 - [ -∞ (worse performance) - 0.0 (no fit) - 1.0 (perfect fit) ]

Process finished with exit code 0
```

## References

- [Kaggle](https://www.kaggle.com/competitions/driving-behaviour)