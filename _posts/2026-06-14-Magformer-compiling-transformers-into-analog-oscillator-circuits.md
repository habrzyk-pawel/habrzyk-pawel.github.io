---
layout: post
title: "Magformer — compiling Transformers into analog oscillator circuits"
date: 2026-06-14 15:55:00 +02:00
---

Over the last few evenings I built a small proof-of-concept called **Magformer** — a software pipeline that takes a frozen PyTorch Transformer and distills it into a network of coupled LC oscillators. The end product is a SPICE netlist that can be simulated in LTSpice or Cadence. The motivation is simple: digital matrix multiplication is expensive, especially for tiny always-on Edge AI (like keyword spotting). If the physics of oscillators can do part of the attention computation for free, maybe we can push inference energy down from hundreds of microjoules to **<10 µJ**.

Repository: [github.com/habrzyk-pawel/magformer](https://github.com/habrzyk-pawel/magformer)

## The big picture

The idea is not new — oscillator-based neural computation has been explored in neuroscience and analog VLSI for decades. What makes this project fun for me is that it treats the problem as a **compiler** problem:

1. Start with a standard Transformer (the *target*).
2. Train an oscillator network to clone its behavior using a fast algebraic approximation.
3. Verify the approximation against real differential equations.
4. Emit fabrication-ready analog component values.

![Magformer pipeline](/assets/images/magformer/pipeline.svg)

## Why analog attention?

A standard self-attention block computes:

<p style="font-size:1.05em;">Attention(<em>Q</em>, <em>K</em>, <em>V</em>) = softmax(<em>QK</em><sup>T</sup> / √<em>d<sub>k</sub></em>) <em>V</em></p>

That means loading three big matrices from DRAM, doing O(n²d) multiplies, and writing the result back. On a small battery-powered sensor this is punishing.

In the analog world, a set of LC tanks connected by resistors naturally follows Kuramoto-style synchronization. The steady-state phase relationships encode something very similar to a similarity/kernel between inputs. If we learn the right frequencies, damping terms, and coupling resistances, the oscillator array can approximate the Transformer's outputs without ever executing a GEMM.

![Digital vs analog inference](/assets/images/magformer/digital-vs-analog.svg)

## The four stages

### 1. Target Transformer

`demo.py` trains a tiny 1-layer Transformer on a toy sequence task. `speech_demo.py` does the real work: it loads the Google Speech Commands ([yes, no, up, down]), extracts MFCCs, and trains a small Transformer that reaches ~89.6% validation accuracy.

### 2. Compiler — Steady-State Approximation

Training a differential equation from scratch in PyTorch is slow, so the compiler uses an algebraic **Steady-State Approximation (SSA)** of the oscillator dynamics. This makes the distillation loop fast enough to run on a laptop, while still capturing the relationship between inputs, frequencies, couplings, and outputs.

### 3. Validator — ODE physics check

`ode_validation.py` takes the learned parameters and integrates the actual continuous-time ODEs with `torchdiffeq`. It then compares the ODE trajectories to the SSA predictions. If they diverge, the learned weights are physically inconsistent and we reject them.

### 4. Exporter — from parameters to resistors

`spice_export.py` maps the abstract oscillator parameters to real 180 nm CMOS components:

- resonant frequency → LC tank values
- damping → parallel resistance
- coupling strength → crossbar resistor

The result is `magformer_chip.cir`, an 8-oscillator prototype netlist.

![Magformer chip prototype](/assets/images/magformer/chip-prototype.svg)

## What works today

- `demo.py` trains end-to-end on synthetic data.
- `speech_demo.py` achieves **100% behavioral cloning agreement** with the golden Transformer on the validation split (the analog prediction matches the digital one class-for-class).
- `ode_validation.py` confirms that the SSA shortcut and the true ODE agree well enough to trust the learned parameters.
- `spice_export.py` produces a valid SPICE netlist with component values in a real process node.

![Oscillator synchronization concept](/assets/images/magformer/oscillator-sync.png)

## Energy estimate

These numbers are back-of-the-envelope, but they show why the idea is worth chasing:

![Energy comparison](/assets/images/magformer/energy-comparison.png)

A keyword-spotting Transformer on a low-power MCU sits around 100 µJ per inference. A correctly sized analog oscillator array doing the same task could theoretically drop below 10 µJ. The exact number depends on clocking, ADC/DAC overhead, and process corners, but the gap is large enough to keep exploring.

## What is missing

This is intentionally a small PoC. Before anyone tapes out a chip, several things need to happen:

1. **Monte Carlo yield simulation** — verify that the learned weights survive ±5% manufacturing variation.
2. **Transistor-level simulation** — run the generated netlist in a real PDK with non-ideal inductors/capacitors.
3. **Noise and temperature analysis** — analog circuits drift; the compiler needs to budget for that.
4. **Better scaling** — 8 oscillators is cute. Hundreds of oscillators, not so cute without careful layout and parasitic extraction.

## Try it

```bash
git clone https://github.com/habrzyk-pawel/magformer.git
cd magformer
pip install torch torchaudio soundfile torchdiffeq
python download_data.py   # fetches Google Speech Commands, ~2.3 GB
python demo.py
python speech_demo.py
python ode_validation.py
python spice_export.py
```

The generated `magformer_chip.cir` can be opened directly in LTSpice.

## Final thoughts

Magformer is the kind of project that sits on the boundary between ML and physics. Most of my recent work has been data-engineering and memory optimization (see the Polars posts), so it was refreshing to think about *where* the computation actually happens. If nothing else, it sharpened my intuition for how much energy we waste moving matrices around in digital hardware.

If you find this interesting, open an issue on the repo — I would love to hear ideas about scaling this beyond 8 oscillators.

---

*Post by Paweł Habrzyk · June 14, 2026*
