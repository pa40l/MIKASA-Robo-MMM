# DSFetch reference

Use the DSFetch implementation already present in
[jezvgg/MIKASA-Robo-MMM, commit 4c5c8b3de941565ee2d57b153a10254fe64c2849](https://github.com/jezvgg/MIKASA-Robo-MMM/tree/4c5c8b3de941565ee2d57b153a10254fe64c2849/robots/fetch).
The robot source, URDF, SRDF and meshes are byte-identical to that revision.
Do not replace this robot with a historical RoboBenchMart revision or stock Fetch.
`reference.json` pins its source, methods and asset hashes. The RoboBenchMart
license notice is retained for attribution of the robot's original source.

The collection profile uses this existing robot with `pd_joint_pos` and 13D
actions. Solver changes and RoboCasa scene collision integration are separate
project code. CabinetSearch applies kitchen exclusions only to wheels/base.

Older collection profiles pinned a different RoboBenchMart revision. Their
recordings and measurements retain that provenance and do not qualify this
profile. A changed robot requires a new physical qualification and run directory.
