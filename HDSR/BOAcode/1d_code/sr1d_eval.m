% sr1d_eval.m
% ===============================================================
% ——Original SR_h1.m; converted script into function for Python call——
% ---------------- 1-D SR full evaluation FUNCTION ---------------
% Input: params1D  = [a, b]   (bistable potential well parameters)
% Output: fr  = performance index (default SNRI, the larger the better)
% ===============================================================

function fr = sr1d_eval(params1D, figure_flag)
%%% ---------- 0.  Read external parameters ----------
a = params1D(1);           % Potential well parameter a
b = params1D(2);           % Potential well parameter b
sigma = 1.4; %        % Noise intensity (fixed)

%%% ---------- 1.  Signal baseline setting ----------
f0 = 0.01;%
Fs = 12;
K  = 5000;
A  = 0.44;%
p  = 0;
x0 = 0;
hsteps = 1/Fs;
t = 0:hsteps:hsteps*K; % Generate an arithmetic sequence with step size h, from 0 to h*K, total elements K+1
t = t(1:end-1); % Remove the last value
Nfft = 2^nextpow2(K);
fbin = Fs*(0:Nfft/2)/Nfft; % Generate frequency array

%%% ---------- 2.  Generate target signal ----------
%S0 = A*cos(2*pi*f0*t' + p);
S0 = A*sin(2*pi*f0*t' + p);%

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

%%% ---------- 4.  Solve SR system ----------
[T,X] = sr1d_solver(t,x0,Fs,S,a,b,K); % Return time vector and state vector
AmpC = max(abs(X))/max(abs(S0)); % Normalization factor
X = X/AmpC;

%%% ---------- 5.  Cross-correlation alignment ----------
[Acor] = xcorr(X,S0);
[~,Ind] = max(abs(Acor));
X = circshift(X,-Ind); % Time-shift alignment

%%% ---------- 7.  Output SNR ----------
NoiPower2 = norm(X' - S0)^2;
SNROut = SigPower/NoiPower2;
SNROut_dB = 10*log10(SNROut);
SNRI = SNROut/SNRIn; % ★ Default optimization target of this function ★
SNRI_dB=10*log10(SNRI);

% === (6) 返回与打印 ===
if figure_flag
    disp('==== result ====');
    disp(['SNRin : ', num2str(SNRIn)]);
    disp(['SNRout: ', num2str(SNROut)]);
    disp(['SNRI  : ', num2str(SNRI)]);
    disp(['SNRin_dB : ', num2str(SNRIn_dB)]);
    disp(['SNRout_dB: ', num2str(SNROut_dB)]);
    disp(['SNRI_dB: ', num2str(SNRI_dB)]);
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
