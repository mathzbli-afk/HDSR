% sr4d_eval.m
% ===============================================================
% ——Original TK.m; converted script into function for Python call—— 
% ---------------- 4-D SR full evaluation FUNCTION ---------------
% Input: params4D  = [a, b, c, ...]   (bistable potential well parameters)
% Output: fr  = performance index (default SNRI, the larger the better)
% ===============================================================
function fr = sr4d_eval(params4D, figure_flag)

%%% ---------- 0. Read external parameters ----------
a = params4D(1);   % Potential well parameter a
b = params4D(2);   % Potential well parameter b
c = params4D(3);   % Potential well parameter c
d = params4D(4);   % Potential well parameter d
e = params4D(5);   % Potential well parameter e
f = params4D(6);   % Potential well parameter f
g = params4D(7);   % Potential well parameter g
h = params4D(8);   % Potential well parameter h
sigma = 1.4;%%can be changed to 2.0       % Noise intensity (fixed)
%%% ---------- 1.  Signal baseline setting ----------
f0 = 0.01;%can be changed to 0.011
Fs = 12;
K  = 5000;
A  = 0.44;%can be changed to 0.5
p  = 0;

hsteps = 1/Fs;
t = 0:hsteps:hsteps*K; % Generate an arithmetic sequence with step size h, from 0 to h*K, total elements K+1
t = t(1:end-1); % Remove the last value
Nfft = 2^nextpow2(K);
fbin = Fs*(0:Nfft/2)/Nfft; % Generate frequency array
%%% ---------- 2.  Generate target signal ----------
%S0 = A*cos(2*pi*f0*t' + p);
S0 = A*sin(2*pi*f0*t' + p);%can be changed to A*cos(2*pi*f0*t' + p);

SigPower = norm(S0)^2;
%%% ---------- 3.  Random noise ----------
randn('seed',7); %%% Fix noise for reproducibility
Nmin = 0;
Nmax = 1;
N = Nmin + Nmax.*randn(K,1); % Generate normally distributed Kx1 random vector, representing white noise
Noi = sigma*N; % Adjust noise intensity by weighting
NoiPower = norm(Noi)^2; % Calculate noise 2-norm

SNRIn = SigPower/NoiPower ; % Input SNR
SNRIn_dB = 10*log10(SNRIn);

S = S0 + Noi;  % Mixed signal

%%% ---------- 4. Solve SR system ----------
Y0 = zeros(4,1);
    [T, Y] = sr4d_solver(t, Y0, Fs, S, a, b, c, d, e, f, g, h, K);
    X = Y(2,:);
    AmpC = max(abs(X))/max(abs(S0)); % Normalization factor
    X = X/AmpC;

%%% ---------- 5. Cross-correlation alignment ----------
[Acor] = xcorr(X,S0);
[~,Ind] = max(abs(Acor));
X = circshift(X,-Ind); % Time-shift alignment

%%% ---------- 7.  Output SNR ----------
NoiPower2 = norm(X' - S0)^2;
SNROut = SigPower/NoiPower2;
SNROut_dB = 10*log10(SNROut);
SNRI = SNROut/SNRIn; % ★ Default optimization target of this function ★
SNRI_dB = 10*log10(SNRI);

% === (6) 返回与打印 ===
if figure_flag
    disp('==== result====');
    disp(['SNRin : ', num2str(SNRIn)]);
    disp(['SNRout: ', num2str(SNROut)]);
    disp(['SNRI  : ', num2str(SNRI)]);
    disp(['SNRin_dB : ', num2str(SNRIn_dB)]);
    disp(['SNRout_dB: ', num2str(SNROut_dB)]);
    disp(['SNRI_dB  : ', num2str(SNRI_dB)]);
end
if figure_flag==2
    %% ---------- 9.  Visualization (separate figures) ----------

    % ===== (1) Target Signal =====
    figure('Position',[100,100,1200,250]);
    plot(t,S0,'r--','LineWidth',1.2)
    xlabel('t/s'); ylabel('s_0');
    xlim([0 420]);
    title(['(a) Target Signal, a=',num2str(a),', b=',num2str(b)]);
    grid on;

    % ===== (2) Noise =====
    figure('Position',[100,100,1200,250]);
    plot(t,Noi,'Color',[0.4 0.4 0.4])
    xlabel('t/s'); ylabel('n');
    xlim([0 420]);
    title(['(b) Noise, a=',num2str(a),', b=',num2str(b)]);
    grid on;

    % ===== (3) Mixed signal =====
    figure('Position',[100,100,1200,250]);
    plot(t,S,'b','LineWidth',1.0); hold on;
    plot(t,S0,'r--','LineWidth',1.2);
    hold off;
    xlabel('t/s'); ylabel('Amplitude');
    xlim([0 420]);
    title(['(c) Mixed vs Original, a=',num2str(a),', b=',num2str(b)]);
    legend({'Mixed S','Original s_0'},'Location','northeast');
    grid on;

    % ===== (4) SR Output =====
    figure('Position',[100,100,1200,250]);
    plot(T,X,'b','LineWidth',1.0);
    xlabel('t/s'); ylabel('x');
    xlim([0 420]);
    title(['(d) SR Output, a=',num2str(a),', b=',num2str(b)]);
    grid on;
end

%fr = SNRI;
fr = SNRI;
end
