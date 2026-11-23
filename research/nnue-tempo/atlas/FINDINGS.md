# Chess findings by reply class

Each row names a concrete class of legal Black replies. Required counts are distinct canonical boards on which forbidding that class for one move worsens Black’s game-theoretic result while another legal reply remains. Erased counts are common-source White move pairs with equal full classical score and different WDL whose ordering disappears or reverses under the restriction in both children. Counts overlap across rows.

SF14.1 is native pair accuracy in percent on the complete erased-pair census. SF18 intact and off are raw large-network accuracies on an independent, one-pair-per-context replication subsample. The final column divides observed WDL-order erasures by the exact expectation from deleting equally many random legal replies from each child on that subsample. Values below one indicate weaker selectivity than the size-matched deletion. A dash marks an empty or undefined estimate.

The [per-rule decision gains](results/findings.csv), [simultaneous channel intervals](results/channel-interventions.csv), [material breakdown](results/complete-rules.csv), [44 move certificates](results/move-certificates.json) and [27 exclusive certificates](results/exclusive-certificates.json) preserve the distinct levels of evidence.

## Captures and exchanges

| Chess test | Required boards | Erased pairs | SF14.1 | Independent pairs | SF18 intact | SF18 off | Erasures / random |
|---|---:|---:|---:|---:|---:|---:|---:|
| The black king takes the white nonking piece. | 698,790 | 10,970,712 | 77.7 | 9,705 | 70.1 | 34.6 | 5.30 |
| The black nonking piece takes its white counterpart. | 5,169,963 | 54,436,458 | 56.9 | 42,047 | 70.2 | 56.4 | 10.90 |
| Black takes the white piece with its man and White can immediately recapture with its king. | 242,954 | 3,251,818 | 59.5 | 2,226 | 93.5 | 58.4 | 8.25 |
| Black takes the white piece with its man and White has no legal king recapture. | 4,927,009 | 51,208,556 | 56.8 | 39,824 | 68.9 | 56.3 | 11.10 |

- **King takes actor** in KNvKP. White’s f2d1 and f2h1 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after f2d1 are c2d1.
- **Clean man capture** in KRvKP. White’s b2c2 and b2a2 receive the same classical score (-224, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after b2c2 are b3c2.

## Promotion and the pawn race

| Chess test | Required boards | Erased pairs | SF14.1 | Independent pairs | SF18 intact | SF18 off | Erasures / random |
|---|---:|---:|---:|---:|---:|---:|---:|
| The black pawn promotes by advancing. | 1,271,495 | 7,016,278 | 97.5 | 2,672 | 97.6 | 88.5 | 2.32 |
| The black pawn promotes by capture. | 36,369 | 809,020 | 42.0 | 370 | 71.8 | 22.0 | 21.12 |
| The black pawn promotes to a rook, bishop or knight. | 481 | 3,477 | 77.3 | 3 | 100.0 | 0.0 | 0.00 |
| The promotion gives immediate check. | 401,633 | 1,389,691 | 93.9 | 627 | 95.5 | 86.9 | 3.42 |
| White has no legal immediate capture of the newly promoted unit. | 1,318,460 | 7,757,875 | 92.0 | 2,981 | 94.5 | 81.8 | 2.95 |
| The black pawn uses its initial two-square advance. | 47,340 | 207,375 | 75.2 | 147 | 75.2 | 76.5 | 0.23 |
| The black pawn advances one square without promotion. | 903,550 | 3,448,864 | 85.5 | 2,483 | 88.8 | 85.2 | 0.65 |
| A nonpromoting pawn advance attacks the surviving white piece. | 17,258 | 70,505 | 88.1 | 129 | 83.3 | 53.9 | 1.11 |
| The black king vacates the square immediately ahead of its pawn. | 522 | 54 | 43.5 | 0 | — | — | 0.00 |
| A king move newly guards its pawn's promotion square. | 45,647 | 399,773 | 79.5 | 350 | 69.4 | 42.4 | 0.62 |
| A king move newly guards the black pawn. | 91,029 | 793,746 | 72.1 | 809 | 71.3 | 47.4 | 0.82 |

- **Quiet promotion** in KRvKP. White’s e2c2 and e2f2 receive the same classical score (-224, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after e2c2 are a2a1q, a2a1r, a2a1b.
- **Surviving promotion** in KNvKP. White’s b3c1 and b3d2 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after b3c1 are b2c1q, b2c1r, b2b1q.
- **Double push** in KPvKP. White’s a2a3 and a2a4 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after a2a3 are g7g5.
- **Single push** in KPvKP. White’s f2f3 and f2f4 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after f2f3 are h6h5.
- **Escort promotion** in KPvKP. White’s h2h3 and h2h4 receive the same classical score (0, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after h2h3 are a3a2.
- **King escorts pawn** in KPvKP. White’s h2h4 and h2h3 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after h2h4 are e7f7, e7f6.

## Checks, double attacks and pins

| Chess test | Required boards | Erased pairs | SF14.1 | Independent pairs | SF18 intact | SF18 off | Erasures / random |
|---|---:|---:|---:|---:|---:|---:|---:|
| The moved nonking piece checks the white king. | 2,582,436 | 19,703,752 | 70.5 | 14,112 | 66.5 | 60.5 | 2.58 |
| A black king move uncovers a check from its own man. | 10,762 | 107,481 | 66.3 | 98 | 57.1 | 52.0 | 1.97 |
| The checking man ends one king step from the white king. | 111,792 | 1,506,183 | 59.9 | 867 | 85.5 | 55.1 | 0.61 |
| The checking man ends at least three king steps from the white king. | 1,216,937 | 10,663,288 | 72.3 | 6,530 | 66.2 | 60.2 | 2.75 |
| The checking man is protected by its king. | 135,016 | 1,284,656 | 66.6 | 879 | 62.4 | 58.7 | 2.20 |
| A checking move leaves the black man geometrically attacking the surviving white piece as well. | 528,870 | 3,312,783 | 75.5 | 3,643 | 64.2 | 63.9 | 2.27 |
| A sliding check lines up the white king in front of its surviving piece. | 411,264 | 5,653,138 | 73.1 | 3,121 | 61.7 | 57.6 | 15.67 |
| The reply creates an absolute pin of the white piece against its king. | 207,015 | 1,980,536 | 69.6 | 1,415 | 58.8 | 58.2 | 12.05 |
| A black king move frees its man from an absolute pin. | 27,640 | 18,078 | 68.9 | 5 | 60.0 | 20.0 | 0.01 |
| An absolutely pinned black man makes a legal move along its pin line. | 198,781 | 1,970,447 | 48.0 | 1,164 | 77.4 | 49.1 | 3.31 |

- **Direct check** in KPvKN. White’s a2a3 and a2a4 receive the same classical score (0, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after a2a3 are h7f6.
- **Discovered check** in KNvKR. White’s c5b7 and c5d7 receive the same classical score (190, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after c5b7 are a7b8, a7b7, a7b6.
- **Skewer** in KRvKR. White’s d7b7 and d7d2 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after d7b7 are a1b1.
- **Unpin** in KRvKP. White’s a4e4 and a4a1 receive the same classical score (-192, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after a4e4 are e2d2.
- **Stay on pin line** in KQvKP. White’s g3h4 and g3f2 receive the same classical score (-2474, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after g3h4 are h3h2.

## Attack, support and counterattack

| Chess test | Required boards | Erased pairs | SF14.1 | Independent pairs | SF18 intact | SF18 off | Erasures / random |
|---|---:|---:|---:|---:|---:|---:|---:|
| The attacked black man moves quietly to a square outside both white attack maps. | 3,187,138 | 4,569,112 | 71.1 | 4,201 | 71.5 | 70.2 | 0.39 |
| A king move newly guards a black man under geometric attack from White. | 63,170 | 119,664 | 66.3 | 46 | 69.6 | 71.7 | 0.06 |
| The black king newly attacks the white piece while guarding its own man. | 14,577 | 70,536 | 77.0 | 59 | 84.7 | 72.9 | 0.44 |
| The black king newly attacks a white piece outside its own king's guard. | 127,808 | 228,289 | 71.1 | 260 | 77.3 | 75.6 | 0.51 |
| The black king newly attacks a white piece guarded by its own king. | 7,224 | 31,342 | 76.1 | 25 | 84.0 | 84.0 | 1.23 |
| Under attack, the black man moves quietly and attacks the surviving white piece. | 312,570 | 1,437,743 | 67.6 | 2,186 | 63.2 | 62.3 | 0.33 |
| The black man newly attacks the white piece outside the white king's guard. | 1,038,441 | 5,548,269 | 74.1 | 5,006 | 64.2 | 62.3 | 3.88 |
| The black man newly attacks the white piece inside the white king's guard. | 86,514 | 355,430 | 64.9 | 555 | 65.8 | 71.6 | 3.07 |
| The black king interposes between its man and an attacking white slider. | 0 | 0 | — | 0 | — | — | — |

- **Escape attack** in KPvKP. White’s f2f3 and f2f4 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after f2f3 are a7a6, a7a5.
- **King rescue** in KNvKR. White’s g8e7 and g8f6 receive the same classical score (133, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after g8e7 are b6c5.
- **King fork defence** in KRvKP. White’s f2f5 and f2g2 receive the same classical score (-192, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after f2f5 are h4g4.
- **King attack loose** in KNvKP. White’s e5f7 and e5d3 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after e5f7 are d5e6.
- **King attack guarded** in KNvKP. White’s c3b1 and c3d1 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after c3b1 are b3c2.
- **Attack loose** in KPvKN. White’s a2a3 and a2a4 receive the same classical score (0, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after a2a3 are a1c2.
- **Attack guarded** in KNvKP. White’s d3c1 and d3e1 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after d3c1 are e2e1q.

## King geometry and pawn control

| Chess test | Required boards | Erased pairs | SF14.1 | Independent pairs | SF18 intact | SF18 off | Erasures / random |
|---|---:|---:|---:|---:|---:|---:|---:|
| A king move establishes rank or file opposition with one intervening square. | 81,425 | 717,053 | 73.3 | 668 | 61.1 | 38.8 | 0.58 |
| A king move establishes diagonal opposition with one intervening square. | 51,887 | 519,624 | 74.0 | 589 | 71.5 | 45.8 | 0.51 |
| A quiet rook or queen move puts a complete attacked rank or file between the two kings. | 26,015 | 14,225 | 69.7 | 88 | 75.6 | 80.1 | 0.02 |
| A king move ends on the rank or file of the white major piece and breaks its complete line cut. | 0 | 0 | — | 0 | — | — | — |
| A black man moves immediately in front of the white pawn. | 17,226 | 106 | 32.1 | 7 | 64.3 | 71.4 | 1.60 |
| The black king steps immediately in front of the white pawn. | 5,144 | 175 | 95.4 | 18 | 97.2 | 83.3 | 5.88 |
| A black man newly attacks the white pawn's promotion square. | 256,750 | 0 | — | 0 | — | — | 0.00 |
| A black man newly attacks the white pawn's next square. | 372,163 | 25 | 88.0 | 1 | 100.0 | 100.0 | 0.19 |
| The black king reduces its distance to the surviving white pawn. | 234,716 | 2,539 | 83.3 | 205 | 87.6 | 84.9 | 2.64 |
| A black pawn push puts its promotion square outside the white king's remaining one-step catch distance. | 58,846 | 262,069 | 79.2 | 206 | 77.7 | 74.0 | 0.29 |
| A black rook or queen newly attacks a white pawn along its file from behind. | 64,022 | 0 | — | 0 | — | — | — |
| The black king moves from outside into the inner four central squares. | 70,874 | 867,140 | 66.9 | 707 | 68.7 | 41.1 | 0.61 |

- **Orthogonal opposition** in KPvKP. White’s h2h4 and h2h3 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after h2h4 are e7e6.
- **Diagonal opposition** in KNvKP. White’s c4d6 and c4b2 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after c4d6 are e2d3.
- **Cut king** in KPvKR. White’s a2a3 and a2a4 receive the same classical score (1373, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after a2a3 are h8h4, h8h3.
- **Attack next push** in KPvKN. White’s a2a3 and a2a4 receive the same classical score (0, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after a2a3 are d1b2.
- **Approach white pawn** in KPvKP. White’s h2h3 and h2h4 receive the same classical score (0, Black perspective). Black obtains a draw and a loss, respectively. Suppressing this reply class changes the two results to loss and loss. The affected optimal replies after h2h3 are b3c4, b3c3.
- **Leave pawn square** in KNvKP. White’s c3a2 and c3b1 receive the same classical score (0, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after c3a2 are h7h6, h7h5.
- **King invasion** in KPvKP. White’s h2h3 and h2h4 receive the same classical score (7, Black perspective). Black obtains a win and a draw, respectively. Suppressing this reply class changes the two results to draw and draw. The affected optimal replies after h2h3 are e6e5.

## Terminal replies

| Chess test | Required boards | Erased pairs | SF14.1 | Independent pairs | SF18 intact | SF18 off | Erasures / random |
|---|---:|---:|---:|---:|---:|---:|---:|
| Black gives immediate checkmate. | 42,899 | 187,500 | 66.3 | 123 | 55.3 | 51.2 | 15.91 |
| Black leaves White stalemated. | 1,081 | 4,652 | 56.7 | 6 | 100.0 | 83.3 | 4.58 |


## Additional distinctions with nine immediate tactical signals matched

Both children have the same nine availability flags for captures, exchanges and promotions. A direction was frozen on discovery pairs under that same constraint. Each row below passed a simultaneous 95% lower bound above 50% on the replication panel. Columns count comparisons where this additional flag differs and give the accuracy of its frozen direction. A minus direction means the presence of a legal reply predicts a worse result for Black. This can reflect the situation that necessitates the reply.

| Available reply | Direction for Black | Replication pairs | Correct % | Simultaneous 95% interval | Played pairs | Played correct % |
|---|---|---:|---:|---:|---:|---:|
| The black pawn advances one square without promotion. | better | 268 | 96.3 | 92.7–99.8 | 40 | 95.0 |
| A king move newly guards its pawn's promotion square. | worse | 61 | 70.5 | 52.4–88.6 | 24 | 33.3 |
| The moved nonking piece checks the white king. | better | 733 | 70.1 | 64.4–75.8 | 8 | 75.0 |
| The checking man ends at least three king steps from the white king. | worse | 1,102 | 73.2 | 67.5–79.0 | 5 | 100.0 |
| The checking man is protected by its king. | worse | 323 | 79.6 | 71.8–87.3 | 5 | 60.0 |
| A checking move leaves the black man geometrically attacking the surviving white piece as well. | better | 4,202 | 91.9 | 90.5–93.2 | 2 | 100.0 |
| A sliding check lines up the white king in front of its surviving piece. | better | 3,020 | 94.0 | 92.7–95.4 | 13 | 100.0 |
| The reply creates an absolute pin of the white piece against its king. | better | 1,935 | 77.4 | 73.4–81.5 | 8 | 100.0 |
| A black king move frees its man from an absolute pin. | worse | 1,293 | 99.2 | 98.5–100.0 | 56 | 89.3 |
| The attacked black man moves quietly to a square outside both white attack maps. | better | 4,606 | 56.4 | 53.9–59.0 | 50 | 62.0 |
| The black king newly attacks the white piece while guarding its own man. | better | 329 | 59.9 | 51.7–68.0 | 26 | 46.2 |
| The black king newly attacks a white piece outside its own king's guard. | better | 2,757 | 60.4 | 57.0–63.7 | 60 | 60.0 |
| Under attack, the black man moves quietly and attacks the surviving white piece. | better | 3,784 | 61.7 | 59.3–64.2 | 10 | 70.0 |
| The black man newly attacks the white piece outside the white king's guard. | better | 5,238 | 60.1 | 58.0–62.3 | 35 | 57.1 |
| The black man newly attacks the white piece inside the white king's guard. | better | 2,113 | 69.3 | 66.3–72.4 | 13 | 84.6 |
| A king move establishes diagonal opposition with one intervening square. | worse | 1,216 | 60.6 | 55.7–65.6 | 35 | 71.4 |
| A quiet rook or queen move puts a complete attacked rank or file between the two kings. | worse | 310 | 61.0 | 51.8–70.1 | 3 | 100.0 |
| A black man moves immediately in front of the white pawn. | better | 95 | 68.4 | 53.4–83.5 | 0 | — |
| The black king reduces its distance to the surviving white pawn. | better | 39 | 79.5 | 57.9–101.1 | 0 | — |

Only the single pawn push and unpin pass the same interval criterion on played positions. In all 1,293 replication unpin comparisons, the black man is already pinned in precisely one child. Among 634 escape comparisons where the black man is attacked in both children, a safe quiet escape predicts the true ordering in 99.2%. Among 3,972 where the initial attack status differs, accuracy is 49.6%. [All 48 matched tests](results/incremental-rules.csv) · [Prerequisite audit](results/prerequisites.csv)
