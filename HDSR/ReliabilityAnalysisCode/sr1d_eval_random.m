 function [SNRI,SNRI_dB, fhat] = sr1d_eval_random(params1D,freq,sigma_in)
% ===============================================================
% 1-D SR evaluation (random noise version, for reliability analysis)
% Single-trial version with random noise (used for reliability statistics)
%
% Inputs:
%   params1D    = [a, b]               % Parameters
%   freq        = Externally provided center frequency (optional)
% Notes:
%   - Fixed random seed randn('seed',7) ensures reproducible noise
%   - f0 can be perturbed externally
% ===============================================================

% Default handling for missing input arguments
if nargin < 2
    f0 = 0.01; % Default center frequency
else
    f0 = freq; % Use external frequency (if analysis.m does not provide perturbation, default to center freq)
end
% --- sigma handling (backward compatible) ---
if nargin < 3 || isempty(sigma_in)
    sigma = 1.4;         % Default noise intensity (original setting)
else
    sigma = sigma_in;    % Use externally provided sigma for MC trials
end

%%% ---------- 0.  Read external parameters ----------
a = params1D(1);           % Potential well parameter a
b = params1D(2);           % Potential well parameter b

%sigma = 1.4;               % Noise intensity (fixed)
%%% ---------- 1.  Signal baseline setup ----------
%f0=0.01
Fs = 12;                           % Sampling frequency (Hz)
K  = 5000;                         % Number of sampling points
A  = 0.44;                         % Signal amplitude
p  = 0;                            % Initial phase
x0 = 0;                            % Initial state
hsteps = 1/Fs;                     % Step size
t = 0:hsteps:hsteps*K;             % Generate time sequence (arithmetic progression with step h)
t = t(1:end-1);                    % Remove the last element
Nfft = 2^20;                       % FFT length (power of 2)
fbin = Fs*(0:Nfft/2)/Nfft;         % Frequency bins (single-sided)

% ---------- 2. Construct target pure signal s(t) ----------
% By default use sine; can switch to cosine if needed (no change to metric definition)
S0 = A * sin(2*pi*f0*t' + p);      % Column vector (K×1)
% ||s||_2: used for SNR definition (can be changed to RMS power if needed)
SigNorm = norm(S0)^2;               % 2-norm (energy scale)

% ---------- 3. Generate random Gaussian white noise n(t) ----------
% N ~ N(0,1), then scaled by sigma: Noi = sigma * N
randn('seed',7);                   % Fixed pseudo-random noise seed (reproducible)
Nmin = 0;
Nmax = 1;
N = Nmin + Nmax.*randn(K,1);       % Standard normal white noise (K×1 vector)

Noi = sigma * N;                   % Noise intensity scaling
NoiNorm = norm(Noi)^2;              % Noise 2-norm

SNRIn = SigNorm/NoiNorm;         % Input SNR (amplitude ratio definition)
SNRIn_dB = 10*log10(SNRIn);       % Input SNR in dB (for debug/printing)

S = S0 + Noi;                      % Mixed signal (s(t) + n(t))

% ---------- 4. Pass through SR nonlinear system ----------
[T,X] = sr1d_solver(t,x0,Fs,S,a,b,K); % Returns time vector and state vector
% Amplitude normalization (avoid distortion in frequency ratio)
AmpC = max(abs(X))/max(abs(S0));   % Normalization factor
X = X/AmpC;

% ---------- 5. Cross-correlation alignment (align x(t) with s(t)) ----------
[Acor] = xcorr(X,S0);
[~,Ind] = max(abs(Acor));
X = circshift(X,-Ind);             % Time shift alignment

% ---------- FFT for XF, estimate frequency peak and convert to Hz ----------
XF = fft(X,Nfft);
XF = abs(XF/K);
XF = XF(1:Nfft/2+1);
XF(2:end-1) = 2*XF(2:end-1);

% % ---------- Estimate frequency (Hz) ----------
[~,IndX] = max(XF);%global maximum,if use window,delete this line
fhat = fbin(IndX);

% ---------- 7. Output SNR and SNRI ----------
NoiNorm2 = norm(X' - S0)^2;
SNROut = SigNorm/NoiNorm2;
SNROut_dB = 10*log10(SNROut);     % Output SNR in dB
SNRI = SNROut/SNRIn;               % Signal-to-Noise Ratio Improvement (default optimization target)
SNRI_dB = 10*log10(SNRI);  
end
