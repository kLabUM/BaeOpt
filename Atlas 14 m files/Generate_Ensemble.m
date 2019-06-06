% Generate the ensemble 
rainfall = [];
for i=1:2000
    rainfall_intensity = rain_pds_a2_sample(11, 5, 1);
    rainfall = [rainfall rainfall_intensity];
end
