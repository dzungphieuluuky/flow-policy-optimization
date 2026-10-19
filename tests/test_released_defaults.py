"""The config defaults must equal FPO++'s released manipulation values."""

from myfpo.configuration import MyfpoConfig

# name -> (dataclass default, released value, where it comes from).
RELEASED = {
    # Isaac Lab ships 16 too (rl_cfg.py:217). Spot and H1 raise it to 32.
    "num_mc_samples": (16, "rl_cfg.py:217 / finetune_online_rl.py:93"),
    "clip_param": (0.01, "finetune_online_rl.py:117 clip_coef"),
    "spo_clip_param": (0.01, "finetune_online_rl.py:118 spo_clip_coef"),
    "num_learning_epochs": (10, "finetune_online_rl.py:74 update_epochs"),
    "num_steps_noise_start": (10, "flow_model_config.py:68 sampling_steps"),
    "num_steps_zero_start": (10, "flow_model_config.py:68 sampling_steps"),
    "policy_lr": (1e-5, "finetune_online_rl.py:124 learning_rate_actor"),
    "critic_lr": (1e-4, "finetune_online_rl.py:125 learning_rate_critic"),
    "adam_betas": ((0.9, 0.99), "finetune_online_rl.py:128 optimizer_betas_actor"),
    "gae_lambda": (0.95, "finetune_online_rl.py:120"),
    "gamma": (0.99, "finetune_online_rl.py:119 discount"),
    "max_grad_norm": (1.0, "finetune_online_rl.py:121"),
    "value_loss_coef": (1.0, "finetune_online_rl.py:121 vf_coef"),
    "normalize_advantage": (True, "finetune_online_rl.py:121 norm_adv"),
    "actor_lr_warmup_steps": (5, "finetune_online_rl.py:131"),
    "critic_lr_warmup_steps": (1, "finetune_online_rl.py:132"),
    # Mechanism defaults, same provenance as the value knobs.
    "do_average_cfm_loss_in_chunk": (False, "finetune_online_rl.py:92"),
    "cfm_average_group_size": (1, "finetune_online_rl.py:95"),
    "cfm_loss_weight_from_t": ("constant", "finetune_online_rl.py:98"),
    "cfm_loss_target": ("velocity", "flow_model_config.py cfm_loss_mode='u'"),
    "actor_arch": ("dit", "no actor arch is configurable; this repo's own"),
    "critic_share_encoder": (False, "always shared upstream"),
    "spo_clip_param_is_separate": (True, "two coefficients, not one"),
}


def test_the_defaults_are_the_released_ones():
    cfg = MyfpoConfig()
    wrong = {
        name: (getattr(cfg, name), want)
        for name, (want, _) in RELEASED.items()
        if name != "spo_clip_param_is_separate"
        and getattr(cfg, name) != want
    }
    assert not wrong, f"drifted from the release: {wrong}"


def test_spo_keeps_its_own_coefficient():
    """FPO++ does not share one clip between PPO and SPO."""
    assert MyfpoConfig().spo_clip_param == 0.01