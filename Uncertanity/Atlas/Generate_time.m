% Generate the ensemble 
for i=1:5
    rainfall_intensity = rain_pds_a2_sample(11, 5, 1);
    dt = 5/60; 
    dur = 24;
    [rain_transform,t]=scs_II_transform(dt,dur,rainfall_intensity);
    rain_transform = [0 diff(rain_transform) 0];
    t = [t t(end) + t(end)-t(end-1)];
    plot(t, rain_transform)
end