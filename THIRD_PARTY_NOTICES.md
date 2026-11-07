# Third-party material

crbot's own code is original. It includes or downloads the following
material from other projects, all under the MIT licence.

| What | Where in crbot | Source |
|---|---|---|
| Card reference art (139 images) and screen reference crops (4 images) | `crbot/assets/cards/`, `crbot/assets/screens/` | [ClashRoyaleBuildABot](https://github.com/Pbatch/ClashRoyaleBuildABot) @ `f89ecbf3`, MIT, © 2022 Peter Batchelor |
| Card and unit metadata (costs, units per card, flying/building) and the detector's class list, extracted from its source | `crbot/assets/cards.json`, `units.json`, `buildabot_classes.json` | ClashRoyaleBuildABot, as above |
| Screen layout measurements (arena grid, hand, elixir bar, tower HP bars, screen-type regions) and HP bar colours | `crbot/geometry.py`, `crbot/vision/hud.py` | ClashRoyaleBuildABot, as above |
| Pretrained unit detector and side classifier, downloaded on demand by `python -m crbot fetch-models` (not stored in this repository) | `models/buildabot/` | ClashRoyaleBuildABot, as above |
| Detection class list | `crbot/assets/katacr_classes.yaml` | [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset), MIT, © 2024 wty |
| Tall-phone layout constants | `crbot/geometry.py` (`LAYOUT_TALL`) | [KataCR](https://github.com/wty-yy/KataCR), MIT |
| Three gameplay frames used as test fixtures | `tests/fixtures/live_*.jpg` | Cropped from [Hasty-CR](https://github.com/hastylmao/Hasty-CR)'s `docs/assets/bot-showcase.mp4`, MIT |

Ideas, but no code, were taken from Hasty-CR (MIT): the separate ally/enemy
classifier, bottom-centre foot points, voting over frames for hand cards,
opponent elixir and cycle inference, and the "towers read full at match start"
check.

Clash Royale and its artwork are trademarks and copyright of Supercell. This
project is not affiliated with or endorsed by Supercell.

---

MIT License (applies to each project above, with its own copyright line)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
