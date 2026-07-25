# MaterialX UDIM cube

Official MaterialX TestSuite UDIM textures (`grid.1001.png` … `grid.1013.png`) plus
`cube.obj` whose UVs span tiles 1001–1013.

## Quick test

```bash
# cube.abc is shipped with UDIM UVs. To rebuild from the OBJ:
sol_obj_to_abc examples/udim_cube/cube.obj examples/udim_cube/cube.abc udim_cube
```

In Sonya_Render:
1. Load `examples/udim_cube/cube.abc`
2. Open the Material Network on the material node
3. Set the image `file` to `…/examples/udim_cube/grid.<UDIM>.png`
   (or paste `udim.mtlx` into the material’s mtlx parameter)

Each face should show a differently numbered grid tile (1001–1013).
