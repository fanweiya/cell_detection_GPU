# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['rmq_main.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=["sklearn","sklearn.utils._cython_blas","sklearn.utils._typedefs","sklearn.neighbors.typedefs","sklearn.neighbors.quad_tree"
			 ,"sklearn.tree._utils","sklearn.tree","sklearn.neighbors._partition_nodes","skimage.restoration._unwrap_1d","skimage.filters.edges"],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='rmq_main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='rmq_main')
