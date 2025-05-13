clear all
close all 

% Import Quadriga library in your path

for config=1:50

    % Parameters
    scenario = 'WINNER_UMa_C2_LOS'; % 'mmMAGIC_UMi_LOS'
    fc = 2.4e9;
    B = 256;
    UE = 16;
    BW = 5e6;

    % Set up simulation parameters
    s = qd_simulation_parameters;
    s.center_frequency = fc;
    lambda = s.wavelength;

    % Create new QuaDRiGa layout
    l = qd_layout(s);
    % UE
    l.no_rx = UE;
    % l.randomize_rx_positions(50, 1.5, 1.5, 1, [1:UE], 50); % Randomize UE positions
    l.randomize_rx_positions; % Randomize UE positions
    l.rx_array = qd_arrayant('omni'); % Omnidirectional antennas
    for ii = 1:l.no_rx
        l.rx_track(ii).no_snapshots = 10; % Number of snapshots of default track 
    end
    % Scenario
    l.set_scenario(scenario);
    % BS
    l.tx_array.generate('omni'); % Omnidirectional antennas
    l.tx_array.no_elements = B;
    BS_pos_mat = zeros(3,l.tx_array.no_elements);
    % Uniform linear array at BS with half wavelength spacing
    for ii=1:l.tx_array.no_elements
        BS_pos_mat(2,ii) = lambda/2*(ii-1) - lambda/2*(l.tx_array.no_elements-1)/2;    
    end
    l.tx_array.element_position = BS_pos_mat;
    l.tx_position(3) = 25; % 25m BS height

    % Plot the layout
    % disp('plotting scenario');
    % l.visualize([],[],0);                                 
    % view(-33, 60);  

    % Generate qd_channel objects
    c = l.get_channels;

    % Extract channel (frequency domain)
    extracted_channel = zeros(UE,B);
    for i=1:UE
        %extracted_channel(i,:) = reshape(c(i,1).coeff(1,:,1,1),[1,B]);
        extracted_channel(i,:) = c(i).fr(BW,1,1);
    end
    file_name = 'WINNER_UMa_C2_LOS_'+string(UE)+'_'+string(B)+ '_' + string(config)+'.csv';
    writematrix(transpose(extracted_channel),file_name);
end
