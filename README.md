# 🌐 Follower Algorithm on a Double-Torus Manifold
*A compact mathematical formulation of phase-coherent attention dynamics*

This repository presents a minimal, self-contained mathematical model of **phase-coherent attention**.  
The model represents attention as a trajectory on a **double-torus Möbius manifold**, driven by the goal of minimizing instantaneous phase mismatch between:

- semantic phase (text)
- perceptual phase (audio / prosody)

The resulting trajectory describes how attention reorganizes itself to resolve semantic–prosodic conflicts and converge toward coherence.

---

## 🧮 Core Formulas of the Model

The entire model is defined by five equations:

<p align="center">
  <img src="Formulas.png" width="520">
</p>

---

## 🧩 Interpretation

- **Formula 1** defines the semantic–perceptual phase variables and the double-torus manifold.  
- **Formula 2** introduces the phase-discrepancy vector field.  
- **Formula 3** defines the Follower’s optimal local motion.  
- **Formula 4** incorporates an energy (effort) constraint into the dynamics.  
- **Formula 5** integrates local motion into a full attention trajectory.

These five expressions together define a smooth, interpretable dynamical system whose trajectories reveal how attention moves through a resonance landscape.

---

## 🧠 Conceptual Motivation

Biological attention is not discrete.  
It is a **continuous phase-alignment process** between:

- low-frequency semantic cycles (left hemisphere)  
- high-frequency perceptual cycles (right hemisphere)

Representing these two cycles as a **double torus** produces a natural geometric model of inter-hemispheric coordination.

The Follower trajectory corresponds to the direction of **maximal instantaneous coherence**, analogous to how the brain minimizes prediction error.

---

## 📜 License
MIT License — see `LICENSE`.

