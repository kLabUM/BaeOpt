function inp = write_rainfall(total_in_rainfall, name)

% Get the timeseries for an SCS II design storm
%  This returns the cumulative rainfall
%  Max recommended storm length is 24 hours
dt = 5/60; 
dur = 24;
[rain_transform,t]=scs_II_transform(dt,dur,total_in_rainfall);

% Convert cumulative rainfall to intensity by taking the difference 
%  between each datapoint
rain_transform = [0 diff(rain_transform) 0];
t = [t t(end) + t(end)-t(end-1)];

% Separate the hours and mins from the time array orginally in decimal hours
hours = floor(t);
t_mins = t - hours;
mins   = round(t_mins * 60);


% Append timeseries to prepared .inp-file template
%  Define the filepaths
%inp = [pwd sprintf('/runfile_xw_a2_template_%06g.inp',n_sample)];
%template = [pwd ];
inp = './'+name+'runfile.inp';
%inp = [pwd sprintf(temp)];
%inp  = [pwd temp];

%  Create the file
copyfile('./runfile_xw_a2_template.inp',inp);
fid = fopen(inp,'a');

%fwrite(fid, sprintf('\n'));

% Append timeseries to prepared .inp-file template
for m = 1:length(rain_transform)
    %fprintf('%s	          	%02d:%02d:%02d  	%f\n','DESIGN_10YR12HR_ALT',hours(m),mins(m),0,rain_transform(m))
    tmp_line = sprintf('%s	          	%02d:%02d:%02d  	%f\n','DESIGN_10YR12HR_ALT',hours(m),mins(m),0,rain_transform(m));
    fwrite(fid,tmp_line);
end

fclose(fid);


if abs( total_in_rainfall - trapz(t/dt,rain_transform) ) > .01
    warning('rain_pds_a2_inp_transform.m: Numerical imprecision for rain_pds(%g,%g)\n',i_duration,i_return_period)
end
cumtrapz(t / dt,rain_transform); % check cumulative rainfall
