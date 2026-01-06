def test_phase2_entrypoint_import():
    import glycoml.phase2 as phase2

    assert hasattr(phase2, "train")
    from glycoml.phase2 import train as train_mod

    assert callable(train_mod)
