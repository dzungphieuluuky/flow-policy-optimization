def test_critic_redesign():
    from myfpo.critic import Critic, Encoder, ActionAffine
    assert Encoder.__name__ == "Encoder"
    assert ActionAffine.__name__ == "ActionAffine"
