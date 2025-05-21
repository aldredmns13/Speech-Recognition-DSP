function fig_path = plot_audio_waves(original_file, filtered_file)
    [x1, fs1] = audioread(original_file);
    [x2, fs2] = audioread(filtered_file);

    t1 = (0:length(x1)-1)/fs1;
    t2 = (0:length(x2)-1)/fs2;

    figure('Visible', 'off');
    subplot(2,1,1);
    plot(t1, x1); title('Original Audio'); xlabel('Time (s)'); ylabel('Amplitude');

    subplot(2,1,2);
    plot(t2, x2); title('Filtered Audio'); xlabel('Time (s)'); ylabel('Amplitude');

    fig_path = 'wave_plot.png';
    saveas(gcf, fig_path);
end
