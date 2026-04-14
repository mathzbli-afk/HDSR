function [SNRI,SNRI_dB,fhat] = sr4d_eval_random(params4D, freq,sigma_in)
% ===============================================================
% 4-D SR evaluation (random noise version, for reliability analysis)
% 4-D stochastic resonance - random noise version 
%
% Input:
%   params4D    = [a, b, c ... h]        % Parameters
%   freq        = externally provided center frequency (optional)
% Note:
%   Random seed rand('seed',7) fixed, f0 perturbation applied, 
%   otherwise identical to eval.m
% ===============================================================

% Input parameter fallback (if not provided)
if nargin < 2
    f0 = 0.01; % Default center frequency
else
    f0 = freq; % External frequency (if analysis.m does not provide frequency perturbation, default center freq is used)
end
% --- sigma handling (backward compatible) ---
if nargin < 3 || isempty(sigma_in)
    sigma = 1.4;         % Default noise intensity (original setting)
else
    sigma = sigma_in;    % Use externally provided sigma for MC trials
end

%%% ---------- 0. Parameter reading ----------
a = params4D(1); b = params4D(2);
c = params4D(3); d = params4D(4);
e = params4D(5); f = params4D(6);
g = params4D(7); h = params4D(8);

%sigma = 1.4;
%%% ---------- 1. Signal baseline setting ----------
%f0 = 0.01;
Fs = 12;                           % Sampling frequency (Hz)
K  = 5000;                         % Number of samples
A  = 0.44;                         % Signal amplitude
p  = 0;                            % Initial phase
%x0 = 0;                            % Initial state
hsteps = 1/Fs;                     % Step size
t = 0:hsteps:hsteps*K;             % Generate arithmetic sequence, step h, total K+1 elements
t = t(1:end-1);                    % Remove last value
Nfft = 2^20;                       % FFT points (power of 2)
fbin = Fs*(0:Nfft/2)/Nfft;         % Frequency array (single-sided positive frequencies)

% ---------- 2. Construct target pure signal s(t) ----------
% s(t) with sine; switch to cos if needed for paper consistency
S0 = A * sin(2*pi*f0*t' + p);      % Column vector (K×1)
% ||s||_2: used for SNR definition 
% (Note: can be changed to sqrt(sum(S0.^2)/K) if power definition is desired)
SigPower = norm(S0)^2;               % 2-norm (magnitude “energy” scale)

% ---------- 3. Generate random Gaussian white noise n(t) ----------
% N ~ N(0,1), then scaled by sigma: Noi = sigma * N
randn('seed',7); %%% Fix noise for reproducibility
Nmin = 0;
Nmax = 1;
N = Nmin + Nmax.*randn(K,1);       % Generate K×1 standard Gaussian white noise
Noi = sigma * N;                   % Set noise intensity
NoiPower = norm(Noi)^2;              % Compute noise 2-norm

SNRIn = SigPower/NoiPower ;        % Input SNR (amplitude ratio definition)
SNRIn_dB = 10*log10(SNRIn);       % In dB (for debug/print)

S = S0 + Noi;                      % Mixed signal  s(t) + n(t)

%%% ---------- 4. Solve SR system ----------
Y0 = zeros(4,1);
[T, Y] = sr4d_solver(t, Y0, Fs, S, a, b, c, d, e, f, g, h, K);
X = Y(2,:);
AmpC = max(abs(X))/max(abs(S0));   % Normalization factor
X = X/AmpC;

% ---------- Cross-correlation alignment (align x(t) with s(t)) ----------
[Acor] = xcorr(X,S0);
[~,Ind] = max(abs(Acor));
X = circshift(X,-Ind);             % Time-shift alignment
% ---------- FFT for XF, estimate frequency peak and convert to Hz ----------
XF = fft(X,Nfft);
XF = abs(XF/K);
XF = XF(1:Nfft/2+1);
XF(2:end-1) = 2*XF(2:end-1);

% % ---------- Estimate frequency (Hz) ----------
[~,IndX] = max(XF);%global maximum,if use window,delete this line
fhat = fbin(IndX);

% ---------- 7. Output SNR_out, SNRI ----------
% SNR_out (linear): ||x||_2 / ||x - s||_2
NoiPower2 = norm(X' - S0)^2;
SNROut = SigPower/NoiPower2;
SNROut_dB = 10*log10(SNROut);     % Output SNR in dB
SNRI = SNROut/SNRIn;               % Signal-to-Noise Ratio Improvement (default optimization target)
SNRI_dB = 10*log10(SNRI);  
end
