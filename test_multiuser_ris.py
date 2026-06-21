import numpy as np, channel_model as cm
np.random.seed(42)
pos,_ = cm.generate_devices()
far = pos[cm.CFG.K_NEAR:cm.CFG.K_SC1]
uav = np.array([437.,584.,108.])
N_MC = 200
K = len(far)

# Scenario A (paper's current claim): each user gets its OWN optimal phase
# Scenario B (physically real, simultaneous): RIS configured for ONE user, all others suffer
# Scenario C: time/freq division — each user served in its own slot with its own phase (realizable)

gains_private = []   # A: per-user private optimum (what the paper reports)
gains_shared  = []   # B: all users under a phase optimized for user 0 only

for _ in range(N_MC):
    G = cm.G_vector(uav)
    # configuration optimized for user 0
    phi0 = cm.opt_phi(G, cm.H_ris_dev(far[0]))
    for i in range(K):
        h_d = cm.chan_scalar(uav, far[i], cm.CFG.BLOCK_DB)
        h_r = cm.H_ris_dev(far[i])
        # A: private optimum
        phi_i = cm.opt_phi(G, h_r)
        h_e_priv = cm.composite(h_d, G, h_r, phi_i)
        gains_private.append(10*np.log10(abs(h_e_priv)**2/max(abs(h_d)**2,1e-40)))
        # B: shared config (phi0) — what user i actually gets when RIS serves user 0
        h_e_shar = cm.composite(h_d, G, h_r, phi0)
        gains_shared.append(10*np.log10(abs(h_e_shar)**2/max(abs(h_d)**2,1e-40)))

gp = np.array(gains_private); gs = np.array(gains_shared)
print(f"Scenario A — per-user PRIVATE optimum (paper's current claim):")
print(f"   mean far-user RIS gain = {gp.mean():+.2f} dB")
print(f"Scenario B — SINGLE shared config (optimized for user 0), 12 users simultaneous:")
print(f"   mean far-user RIS gain = {gs.mean():+.2f} dB")
print(f"   => simultaneous-service penalty = {gp.mean()-gs.mean():.2f} dB lost")
print(f"   fraction of users with >1 dB gain under shared config: {(gs>1).mean()*100:.1f}%")
