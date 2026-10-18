"""Exercise archive ingestion without downloading the Yelp dataset."""

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ingestion", Path(__file__).resolve().parents[1] / "scripts" / "00_data_ingestion.py"
)
assert spec is not None
assert spec.loader is not None
ingestion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingestion)


def make_archive(path, omit=None):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as dataset:
        contents = {
            name: (json.dumps(dict.fromkeys(fields, 'example')) + '\n').encode()
            for name, fields in ingestion.EXPECTED_FILES.items()
            if name != omit
        }
        contents[ingestion.AGREEMENT] = b'%PDF-1.3 example'
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            dataset.addfile(member, io.BytesIO(data))
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ingestion.INNER_ARCHIVE, payload.getvalue())
        archive.writestr(ingestion.DOCUMENTATION, b'%PDF-1.3 documentation')
        archive.writestr('__MACOSX/Yelp JSON/._yelp_dataset.tar', b'metadata')


def test_download_sends_required_headers_and_saves_valid_archive(tmp_path, monkeypatch):
    source = tmp_path / 'fixture.zip'
    make_archive(source)
    payload = source.read_bytes()

    class Response:
        headers = {'Content-Length': str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield payload

    def get(url, **kwargs):
        assert url == ingestion.DATASET_URL
        assert kwargs['stream'] is True
        assert kwargs['headers']['User-Agent'] == 'Mozilla/5.0'
        assert (
            kwargs['headers']['Referer'] == 'https://business.yelp.com/data/resources/open-dataset/'
        )
        return Response()

    monkeypatch.setattr(ingestion.requests, 'get', get)
    destination = tmp_path / 'Yelp-JSON.zip'
    ingestion.download(ingestion.DATASET_URL, destination)
    assert destination.read_bytes() == payload
    assert not destination.with_suffix('.zip.part').exists()


def test_nested_release_extracts_sources_and_preserves_existing_files(tmp_path):
    archive = tmp_path / 'Yelp-JSON.zip'
    make_archive(archive)
    raw = tmp_path / 'raw'
    ingestion.extract_archive(archive, raw)
    assert {p.name for p in raw.iterdir()} == set(ingestion.EXPECTED_FILES) | {
        ingestion.AGREEMENT,
        Path(ingestion.DOCUMENTATION).name,
    }
    before = {p.name: p.stat().st_mtime_ns for p in raw.iterdir()}
    ingestion.extract_archive(archive, raw)
    assert before == {p.name: p.stat().st_mtime_ns for p in raw.iterdir()}


def test_rejects_missing_inner_source(tmp_path):
    archive = tmp_path / 'Yelp-JSON.zip'
    make_archive(archive, omit='yelp_academic_dataset_user.json')
    with pytest.raises(ValueError, match='missing.*yelp_academic_dataset_user'):
        ingestion.extract_archive(archive, tmp_path / 'raw')


def test_rejects_wrong_outer_layout(tmp_path):
    archive = tmp_path / 'wrong.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('unexpected.txt', 'example')
    with zipfile.ZipFile(archive) as source, pytest.raises(KeyError):
        ingestion.validate_archive(source)


@pytest.mark.parametrize('already_downloaded', [False, True])
def test_default_archive_uses_raw_directory_only(tmp_path, monkeypatch, already_downloaded):
    # A reference ZIP at the repo root must never bypass the normal download.
    (tmp_path / 'Yelp-JSON.zip').write_bytes(b'reference only')
    raw = tmp_path / 'data' / 'raw'
    raw.mkdir(parents=True)
    expected = raw / 'Yelp-JSON.zip'
    if already_downloaded:
        expected.write_bytes(b'previous download')
    downloads = []
    extractions = []
    monkeypatch.setattr(ingestion, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        ingestion,
        'load_config',
        lambda _: {
            'data': {
                'raw_dir': raw,
                'business_file': 'yelp_academic_dataset_business.json',
                'review_file': 'yelp_academic_dataset_review.json',
            }
        },
    )
    monkeypatch.setattr(ingestion, 'download', lambda url, path: downloads.append((url, path)))
    monkeypatch.setattr(
        ingestion, 'extract_archive', lambda path, target: extractions.append((path, target))
    )
    monkeypatch.setattr('sys.argv', ['00_data_ingestion.py'])
    monkeypatch.chdir(tmp_path.parent)
    ingestion.main()
    assert downloads == ([] if already_downloaded else [(ingestion.DATASET_URL, expected)])
    assert extractions == [(expected, raw)]


def test_validation_checks_records_beyond_first_line(tmp_path):
    source = tmp_path / 'reviews.json'
    source.write_text("{'review_id': 'r1'}\nnot json\n", encoding='utf-8')
    with pytest.raises(ValueError, match='line 2'):
        ingestion.validate_jsonl(source, {'review_id'})


def test_rejects_empty_or_wrong_schema(tmp_path):
    source = tmp_path / 'reviews.json'
    source.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='empty'):
        ingestion.validate_jsonl(source, {'review_id'})
    source.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='expected object'):
        ingestion.validate_jsonl(source, {'review_id'})
