# Stage 08 — Implementation
`render_6dof_pose.py <pipeline_result.pkl.gz> <frames_dir> <out_dir>`
Renders the oriented 3D box + pose axes (from the pipeline's mask+depth PCA pose)
and exports `object_6dof_poses.json`. FoundationPose weights dir reserved at
`../../models/foundationpose/` for the CAD-anchored upgrade path.
