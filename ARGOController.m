r = 0.1;
b = 0.6;
w_max = 12;
time_step = 0.02;
sims_duration = 30;
motor_time_constant = 0.25;
noise = 0.10;
max_motor_command = 1.0; 
initial_speeds = 0; 

%build the time matrix (0-30s)
t = 0: time_step: sims_duration; 

%rover command empty vector 
lin_v = zeros(1, length(t)); 
yaw = zeros(1, length(t)); 

for i = 1:length(t)
    if t(i) >= 3 & t(i) < 8
        lin_v(i) = 0.5; 
    elseif t(i) >= 8 & t(i) < 12
        lin_v(i) = 0.4; 
        yaw(i) = 0.5; 
    elseif t(i) >= 12 & t(i) < 16
        lin_v(i) = 0.4;   
        yaw(i) = -0.5; 
    elseif t(i) >= 16 & t(i) < 20
        yaw(i) = 0.8; 
    elseif t(i) >= 20 & t(i) < 24
        lin_v(i) = 1.5; 
    elseif t(i) >= 24 & t(i) < 27
        lin_v(i) = -0.3; 
    end 
end 

w_left = (lin_v - (b/2)*yaw)/r; 
w_right = (lin_v + (b/2)*yaw)/r; 

w_right = max(min(w_right, w_max), -w_max);
w_left  = max(min(w_left,  w_max), -w_max);
w_ref = [w_right; w_left];

%assuming u = 1 
K = w_max/max_motor_command; 

%calculate PID
lambda = 0.1; %ask about this number 
Kp = motor_time_constant/(K*lambda);
Ki = 1/(K*lambda);
Kd = 0;

%setup for sim
w_act = zeros(2, length(t)); 
w_meas = zeros(2, length(t)); 
u = zeros(2, length(t));
I = [0; 0]; 

for j = 1:(length(t)-1)
    w_meas(:,j) = w_act(:,j) + noise*randn(2,1);
    e = w_ref(:, j) - w_meas(:, j); 
    
    u_raw = Kp*e + Ki*I;
    u(:,j) = max(min(u_raw, max_motor_command), -max_motor_command);

    pushing = (abs(u_raw) > max_motor_command) & (sign(u_raw) == sign(e));
    I = I + time_step * e .* ~pushing;
    
    w_act(:,j+1) = w_act(:,j) + (time_step/motor_time_constant) * (K*u(:,j) - w_act(:,j));
end 

w_meas(:,length(t)) = w_act(:,length(t)) + noise*randn(2,1);  
v_act   = r*(w_act(1,:) + w_act(2,:))/2;
yaw_act = r*(w_act(1,:) - w_act(2,:))/b;

figure;
subplot(3,1,1); plot(t, w_ref(1,:), '--', t, w_act(1,:)); title('Right wheel'); ylabel('rad/s'); legend('ref','actual');
subplot(3,1,2); plot(t, w_ref(2,:), '--', t, w_act(2,:)); title('Left wheel'); ylabel('rad/s');
subplot(3,1,3); plot(t, u(1,:), t, u(2,:)); title('Motor command'); ylabel('u'); xlabel('t (s)'); legend('R','L');

figure;
subplot(2,1,1); plot(t, lin_v, '--', t, v_act); title('Linear velocity'); ylabel('m/s'); legend('cmd','actual');
subplot(2,1,2); plot(t, yaw, '--', t, yaw_act); title('Yaw rate'); ylabel('rad/s'); xlabel('t (s)');

sensor_noise = w_meas - w_act;
wheel_error  = w_act  - w_ref;

figure('Name','Sensor Noise vs Controlled Sensor Noise');

subplot(2,2,1); 
plot(t, sensor_noise(1,:)); 
title('Right: sensor noise'); 
ylabel('rad/s');

subplot(2,2,2); 
plot(t, sensor_noise(2,:)); 
title('Left: sensor noise');

subplot(2,2,3); 
plot(t, wheel_error(1,:));  
title('Right: controlled noise'); 
xlabel('t (s)'); 
ylabel('rad/s');

subplot(2,2,4); 
plot(t, wheel_error(2,:));  
title('Left: controlled noise');  
xlabel('t (s)');