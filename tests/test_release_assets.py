"""Exact asset transfer security checks; no research inference or training."""
import hashlib,json,stat,zipfile
from pathlib import Path
import pytest
from p57.release_assets import safe_extract,verify_tree,safe_name,install,expected_bundle,encoded,authority
ROOT=Path(__file__).resolve().parents[1]


def archive(tmp_path,entries):
 p=tmp_path/'fixture.zip'
 with zipfile.ZipFile(p,'w') as z:
  for name,value,mode in entries:
   info=zipfile.ZipInfo(name);info.external_attr=mode<<16;z.writestr(info,value)
 return p


def expected():return {'core/x.txt':{'sha256':hashlib.sha256(b'exact').hexdigest()}}


def test_extract_exact_bytes_and_refuse_occupied(tmp_path):
 z=archive(tmp_path,[('core/x.txt',b'exact',stat.S_IFREG|0o644)])
 out=tmp_path/'out';assert safe_extract(z,out,expected())['all_sha256_match']
 assert (out/'core/x.txt').read_bytes()==b'exact'
 with pytest.raises(FileExistsError):safe_extract(z,out,expected())


@pytest.mark.parametrize('name',['../escape','/absolute','x/../escape','x//double','./dot','x\\evil','C:/evil',''])
def test_reject_unsafe_member_names(name):
 with pytest.raises(ValueError):safe_name(name)


@pytest.mark.parametrize('case',['extra','missing','duplicate','symlink','bad_hash'])
def test_archive_rejection_is_atomic(tmp_path,case):
 entries=[('core/x.txt',b'exact',stat.S_IFREG|0o644)]
 if case=='extra':entries.append(('extra.txt',b'extra',stat.S_IFREG|0o644))
 if case=='missing':entries=[]
 if case=='duplicate':entries*=2
 if case=='symlink':entries=[('core/x.txt',b'/etc/passwd',stat.S_IFLNK|0o777)]
 if case=='bad_hash':entries=[('core/x.txt',b'corrupted',stat.S_IFREG|0o644)]
 with pytest.warns(UserWarning) if case=='duplicate' else __import__('contextlib').nullcontext():z=archive(tmp_path,entries)
 with pytest.raises(ValueError):safe_extract(z,tmp_path/'out',expected())
 assert not (tmp_path/'out').exists()
 assert not (tmp_path/'escape').exists()


def test_reject_linked_destination_and_extra_tree_files(tmp_path):
 z=archive(tmp_path,[('core/x.txt',b'exact',stat.S_IFREG|0o644)])
 real=tmp_path/'real';real.mkdir();(tmp_path/'link').symlink_to(real,target_is_directory=True)
 with pytest.raises(ValueError):safe_extract(z,tmp_path/'link/out',expected())
 safe_extract(z,tmp_path/'out',expected());(tmp_path/'out/extra').write_text('extra')
 with pytest.raises(ValueError):verify_tree(tmp_path/'out',expected())


def test_installer_rejects_wrong_archive_before_writes(tmp_path):
 p=tmp_path/'bad.zip';p.write_bytes(b'not canonical')
 with pytest.raises(ValueError,match='Archive SHA256'):install(tmp_path,p,{'archive_sha256':'0'*64},{})


def test_committed_inventory_matches_authority():
 entries,_=expected_bundle(ROOT)
 inv=json.loads((ROOT/'docs/release_v1/asset_inventory.json').read_text());receipt=json.loads((ROOT/'docs/release_v1/assets_receipt.json').read_text())
 assert hashlib.sha256(encoded(inv)).hexdigest()==receipt['inventory_sha256']
 assert set(inv['files'])==set(entries) and len(entries)==184
 for name,row in entries.items():assert row['sha256']==inv['files'][name]['sha256']
 m=authority(ROOT)
 assert len(m['splits']['val'])==len(m['splits']['test'])==45
