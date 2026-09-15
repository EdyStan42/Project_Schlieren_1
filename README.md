# Fluid Density Analysis through Schlieren Imaging

## Overview
This repository contains the methodology and theoretical foundation for analyzing small fluid density variations using Schlieren imaging techniques and Python. By exploiting the changes in the refractive index of a fluid caused by variations in temperature, speed, or chemical composition, this project visualizes and calculates the velocity of heated air streams (e.g., from a candle). 

For a comprehensive theoretical and experimental breakdown, please refer to the primary documentation file: **Analiza densitatii fluidelor prin imagistica Schlieren.pdf**.

## Methods & Implementation

The project explores two primary techniques for capturing refractive index variations:

*   **Classic Schlieren Imaging:** A traditional, non-intrusive optical setup utilizing a camera, a point light source, a spherical mirror, and an obstacle (e.g., a thin wire). The mirror is placed at a distance equal to its radius of curvature, calculated as $R=2f$. The density variations act as thin lenses (defined by the ideal gas density relation $\rho=\frac{M P}{R T}$), refracting light around the obstacle to create visible gradients.
*   **Background Oriented Schlieren (BOS):** A modern, highly accessible technique replacing the mirror with a patterned background (printed or digital). This method relies heavily on computational image processing to track the apparent distortion of the background caused by the fluctuating fluid density in the foreground.

### Python Image Processing Pipeline (BOS)
The software tracks the optical flow and calculates fluid velocity using the following pipeline:
1.  **Comparison Strategy:** Compares the current frame against a fixed reference frame (for precise density visualization) or adjacent frames (for qualitative velocity measurement).
2.  **Region of Interest (ROI):** Defines a narrow processing area to accurately capture vertical pixel fluctuations.
3.  **Velocity Calculation:** Measures the speed of pixel intensity fluctuations and converts it to real-world velocity using the frame rate: $V=v_{f}F_{ps}$. To eliminate hardware-induced refresh rate noise, a secondary control region is used to subtract baseline fluctuations.
4.  **Enhancement:** Applies Contrast Limited Adaptive Histogram Equalization (CLAHE), gamma correction, and boundary masking to isolate the gas flow from the heat source visually.

## Theoretical Verification

To validate the experimental optical flow measurements, the software's output is compared against mathematical fluid dynamics models. 

For the velocity of air heated by a point source at height $z$, we use the Gaussian approximation:

$$u_{c}(z)=C\left[\frac{g\beta Q}{\rho_{0}C_{p}T_{0}}z\right]^{-1/3}$$

*Experimental vs. Theoretical Results:*
*   **Theoretical Speed (at 2cm above source):** 0.298 m/s
*   **Measured Speed (Software):** ~0.37 m/s
*   **Error Rate:** ~24.16% 

Using the calculated velocity, the Reynolds number at the candle's exit is determined to verify the flow regime (transitional flow between laminar and turbulent):

$$R_{e}=\frac{u_{c}(z)\cdot b(z)}{\mu}$$

## Future Work
*   Integration of Artificial Intelligence (AI) to better isolate the gas flow from complex backgrounds.
*   Hardware upgrades (larger, more sensitive camera sensors) to reduce visual noise.
*   Exploring applications in astrophysics, such as modeling light refraction caused by gravitational fields (gravitational lensing).

---
*Authors: Stan George-Edward, Radu Sebastian Hristu*
