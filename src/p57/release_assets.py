"""W8B exact asset transfer; standard library only, no inference or downloads."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

MAX_BYTES=256*1024*1024


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '\\' in name or ':' in name or any(x in ('','..','.') for x in name.split('/')):
        raise ValueError('Unsafe archive/path member: '+name)
    return name


def no_links(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('Linked paths rejected: '+str(path))
    return path


def authority(root):
    root=Path(root);cfg=json.loads((root/'configs/reproduction_v1.json').read_text())
    path=root/cfg['manifest']
    if sha(path)!=cfg['manifest_sha256']:raise ValueError('Release manifest drift')
    return json.loads(path.read_text())


def payload(root):
    m=authority(root);entries={}
    for split,rows in m['splits'].items():
        for row in rows:
            for field in ('image','label'):
                name='core/'+safe_name(row[field+'_path'])
                entries[name]={'sha256':row[field+'_sha256'],'kind':field,'split':split,'source_id':row['source_id'],
                               'label_text':row['label_text']}
    for key,model in m['checkpoints'].items():entries['models/'+key+'.pt']={'sha256':model['sha256'],'kind':'checkpoint','model':key}
    return m,entries


def expected_bundle(root):
    m,entries=payload(root)
    notices=(Path(root)/'docs/release_v1/ASSET_NOTICES.md').read_bytes()
    provenance=encoded({'release_manifest_sha256':sha(Path(root)/'docs/reproduction_v1/release_manifest.json'),
                        'source':m['source'],'classes':m['classes'],'seed':m['seed'],
                        'splits':m['splits'],'checkpoints':m['checkpoints'],
                        'scope':'90 val/test originals and labels; two canonical checkpoints; no GTSRB/fonts'})
    metadata={'NOTICES.md':notices,'PROVENANCE.json':provenance}
    for name,data in metadata.items():entries[name]={'sha256':hashlib.sha256(data).hexdigest(),'kind':'metadata'}
    return entries,metadata


def verify_tree(directory,expected):
    directory=no_links(directory)
    actual=set()
    for p in directory.rglob('*'):
        if p.is_symlink():raise ValueError('Linked package member')
        if p.is_file():actual.add(p.relative_to(directory).as_posix())
        elif not p.is_dir():raise ValueError('Nonregular member')
    if actual!=set(expected):raise ValueError('Missing/extra package files')
    for name,record in expected.items():
        if sha(directory/safe_name(name))!=record['sha256']:raise ValueError('Asset checksum mismatch: '+name)
    return {'files':len(actual),'bytes':sum((directory/p).stat().st_size for p in actual),'all_sha256_match':True}


def build(root,destination):
    root=Path(root);destination=no_links(destination)
    if destination.exists():raise FileExistsError(destination)
    archive=no_links(destination.with_suffix('.zip'))
    if archive.exists():raise FileExistsError(archive)
    entries,metadata=expected_bundle(root);m=authority(root)
    local=json.loads((root/'configs/reproduction_assets.json').read_text())
    sources={}
    for name,record in entries.items():
        if record['kind']=='metadata':continue
        source=Path(local['core'])/name.removeprefix('core/') if name.startswith('core/') else Path(local['checkpoints'][record['model']])
        source=no_links(source if source.is_absolute() else root/source)
        if not source.is_file() or sha(source)!=record['sha256']:raise ValueError('Missing/corrupt canonical source: '+name)
        sources[name]=source
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='p57-build-',dir=destination.parent) as temp:
        stage=Path(temp)/'package';stage.mkdir()
        for name,source in sources.items():
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for name,data in metadata.items():(stage/name).write_bytes(data)
        inventory={'version':'w8b_assets_v1','classes':m['classes'],'payload_images':90,'payload_labels':90,'checkpoints':2,
                   'files':{name:{**record,'bytes':(stage/name).stat().st_size} for name,record in entries.items()}}
        (stage/'INVENTORY.json').write_bytes(encoded(inventory))
        full={**entries,'INVENTORY.json':{'sha256':sha(stage/'INVENTORY.json')}}
        verify_tree(stage,full)
        stage.rename(destination)
    archive=destination.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(full):
            info=zipfile.ZipInfo(safe_name(name),date_time=(2026,1,1,0,0,0));info.external_attr=(stat.S_IFREG|0o644)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            with (destination/name).open('rb') as source,z.open(info,'w') as target:shutil.copyfileobj(source,target)
    checksum=sha(archive);archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n')
    return {'archive':str(archive),'archive_sha256':checksum,'archive_bytes':archive.stat().st_size,
            'inventory_sha256':sha(destination/'INVENTORY.json'),**verify_tree(destination,full)}


def safe_extract(archive,destination,expected):
    """Preflight ALL names/types/membership before streaming to an isolated new tree."""
    destination=no_links(destination)
    if destination.exists():raise FileExistsError(destination)
    with zipfile.ZipFile(no_links(archive)) as z:
        members=z.infolist();names=[safe_name(i.filename) for i in members]
        if len(names)!=len(set(names)) or len(names)!=len(set(n.casefold() for n in names)):raise ValueError('Duplicate archive members')
        if set(names)!=set(expected):raise ValueError('Missing/extra archive members')
        if sum(i.file_size for i in members)>MAX_BYTES:raise ValueError('Archive size limit')
        for i in members:
            mode=i.external_attr>>16
            if i.is_dir() or (stat.S_IFMT(mode) not in (0,stat.S_IFREG)) or i.flag_bits&1:raise ValueError('Unsafe archive member type')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='p57-extract-',dir=destination.parent) as temp:
            stage=Path(temp)/'assets';stage.mkdir()
            for info in members:
                target=stage/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256();size=0
                with z.open(info) as source,target.open('xb') as out:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        size+=len(block)
                        if size>info.file_size:raise ValueError('Expanded size mismatch')
                        h.update(block);out.write(block)
                if h.hexdigest()!=expected[info.filename]['sha256']:raise ValueError('Asset checksum mismatch: '+info.filename)
            report=verify_tree(stage,expected);stage.rename(destination)
    return report


def install(root,archive,receipt,inventory):
    """Install into existing default-map paths; never modify a config or occupied file."""
    root=no_links(root);archive=no_links(archive)
    if sha(archive)!=receipt['archive_sha256']:raise ValueError('Archive SHA256 mismatch')
    entries,_=expected_bundle(root)
    # Inventory is pinned separately in the committed receipt, never trusted from ZIP alone.
    if hashlib.sha256(encoded(inventory)).hexdigest()!=receipt['inventory_sha256']:raise ValueError('Inventory SHA256 mismatch')
    if set(inventory['files'])!=set(entries):raise ValueError('Inventory membership drift')
    for name,record in entries.items():
        if inventory['files'][name]['sha256']!=record['sha256']:raise ValueError('Inventory authority drift')
    expected={**entries,'INVENTORY.json':{'sha256':receipt['inventory_sha256']}}
    local=json.loads((root/'configs/reproduction_assets.json').read_text());targets={}
    for name,record in entries.items():
        if name.startswith('core/'):relative=Path(local['core'])/name.removeprefix('core/')
        elif record['kind']=='checkpoint':relative=Path(local['checkpoints'][record['model']])
        else:continue
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Installer requires safe relative default asset paths')
        target=no_links(root/relative)
        if target.exists():raise FileExistsError('Refusing occupied asset target: '+str(target))
        targets[name]=target
    with tempfile.TemporaryDirectory(prefix='p57-install-') as temp:
        source=Path(temp).resolve()/'verified';report=safe_extract(archive,source,expected)
        # Validate first; installation uses exclusive writes and keeps explicit failures visible.
        for name,target in targets.items():
            target.parent.mkdir(parents=True,exist_ok=True)
            with (source/name).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
            if sha(target)!=entries[name]['sha256']:raise ValueError('Installed asset checksum mismatch')
    return {**report,'installed_payload_files':len(targets),'config_unchanged':True}
