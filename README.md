# Kalman Filter
This is a simple linear Kalman filter implementation.

## Usage
The filter allows for different measurements to run partial updates as well as full updates using the whole of the measurement vector.
Control is optional.
User needs to structure the measurement vector with the different possible measurements and also create lambdas that create the state transition matrix and the process noise covariance matrix (along with the control matrix if applicable) based on an time input (this being a variable dt).
Pass the measurement vector and the matrix wrappers to the filter instantiation, initialize the filter with an initial state and an initial covariance matrix and the call the update functions upon new measurements.
