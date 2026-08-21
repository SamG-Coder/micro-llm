# Pack id is global 0..47

Upstream `docs/PRUNE_TABLE.md` says:

> `pack` ? Pack id in the 4-layer group

That is a doc bug. A 4-layer group has three DeltaNet packs. Numbering them
`0..2` *inside each group* collides: every group would write pack 0, 1, 2 and
Export could not tell which pack to drop.

## Contract (this tree)

- There are **48** DeltaNet packs.
- `pack` is a **global** id in `0..47`.
- A pack may also be named `(group, slot)` with `group in 0..15`, `slot in 0..2`.
- Layer of that pack: `layer = 4 * group + slot`
  (layers `3,7,11,...,63` are Gated Attention, not packs).
- Global id: `pack = 3 * group + slot`

Examples:

| group | slot | pack | layer |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 0 | 2 | 2 | 2 |
| 1 | 0 | 3 | 4 |
| 15 | 2 | 47 | 62 |

`n_spike > 0` means the pack is not dead, even if the hour average looks like
identity. Dead = `n_spike == 0`.

The on-disk prune table writes packs as a 48-long array in pack-id order.
Each record stores both `pack` (0..47) and `layer` (`4*group+slot`) so Export
does not have to re-derive the map.
