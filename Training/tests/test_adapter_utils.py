from Training.tools import adapter_utils


def test_load_model_forwards_local_files_only(monkeypatch):
    calls = {}

    def fake_loader(name_or_path, **kwargs):
        calls['name'] = name_or_path
        calls['kwargs'] = kwargs
        return 'dummy-model'

    # Ensure adapter path is not used so the fake loader executes
    monkeypatch.setenv('USE_ADAPTER', '0')

    result = adapter_utils.load_model(
        'some-model',
        loader=fake_loader,
        local_files_only=True,
        custom_flag='sentinel',
    )

    assert result == 'dummy-model'
    assert calls['name'] == 'some-model'
    assert calls['kwargs']['local_files_only'] is True
    assert calls['kwargs']['custom_flag'] == 'sentinel'
