# A.R.G.O. III

**A.R.G.O. — Autonomous Rover for Ground Operations**

A.R.G.O. is an autonomous ground rover designed to navigate unknown terrain using onboard perception, localization, planning, and closed-loop control.

## Semester Goal

Build a rover that can:

* Be manually controlled
* Estimate its motion
* Detect nearby obstacles and terrain
* Plan a safe path
* Autonomously navigate a short outdoor course

## Planned Software Stack

* ROS 2
* Python / C++
* RViz
* Gazebo
* OpenCV
* Nav2
* Git / GitHub

## Repository Structure

```text
src/        ROS 2 packages
docs/       Architecture, interfaces, setup, and decisions
config/     Shared configuration
scripts/    Development utilities
tests/      System/integration tests
```

## Development Workflow

1. Create or select a GitHub issue.
2. Create a feature branch.
3. Develop and test locally.
4. Commit changes.
5. Push your branch.
6. Open a pull request.
7. Get at least one teammate review.
8. Merge into `main`.

Do not develop directly on `main`.

## Current Status

Initial software architecture and project setup.

A first terrain classifier (flat / rocky / non-ground from a single photo) is in `src/argo_perception`. It has hand-set rules and an optional model trained on the public RUGD dataset; neither has been checked on photos of our own course. See `src/argo_perception/README.md`.

