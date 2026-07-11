-- The compute kernel inside glissando's L-BFGS smoothing-parameter
-- optimization (Phase 3 benchmark kernel).
--
-- glissando (real-limoges/glissando) fits GAMLSS models; its REML path
-- selects smoothing parameters by running L-BFGS over rho = log
-- lambda (src/fitting/solver.rs, run_optimization_reml). The L-BFGS
-- direction update itself is O(m*k) for k = #penalties (tiny); the
-- compute that dominates every iteration is the cost/gradient
-- evaluation, ported here from:
--
--   fit_pwls_with_grad_info (solver.rs:479):
--     x_weighted = sqrt(W) . X        (row scaling)
--     X'WX = x_weighted' x_weighted   (the big matmul)
--     X'Wz = x_weighted' (sqrt(W) . z)
--     fitted = X beta; r = z - fitted; rss = sum(w r^2)
--     X'Wr = x_weighted' (sqrt(W) . r)
--   RemlCost::gradient (solver.rs:239):
--     per penalty j: bsb    = beta' S_j beta
--                    tr_v   = sum(V * S_j)        (Hadamard sum)
--                    tr_pinv= sum(S_pinv * S_j)   (Hadamard sum)
--                    grad_j = -0.5 lambda_j (-bsb + tr_pinv - tr_v)
--
-- The dense solve/inverse of (X'WX + S_lambda) stays in LAPACK in
-- glissando and is out of scope (not in the map/reduce/scan subset);
-- V and S_pinv are inputs here, as the gradient formula treats them.
--
-- NOTE: glissando is f64 throughout. Apple GPUs have no f64, so this
-- port is f32 *for all three benchmarked backends* — the comparison
-- stays apples-to-apples, but absolute numerics differ from glissando.
-- CI-sized copy of heavy-metal's bench/kernel.fut (keep in sync).
--
-- The cross-backend comparison tests the 1/n-scaled entry: futhark's
-- output comparison widens its tolerance only with *positive* expected
-- values (futhark-data Compare.tolerance has no abs), so raw
-- large-magnitude negative outputs (X'Wr ~ -4e3) face a 0.002
-- *absolute* tolerance that f32 reduction reassociation cannot meet.
-- Scaling makes the comparison effectively relative.

def dotprod (xs: []f32) (ys: []f32) : f32 =
  reduce (+) 0 (map2 (*) xs ys)

def matvec [n][p] (m: [n][p]f32) (v: [p]f32) : [n]f32 =
  map (`dotprod` v) m

-- X'WX and X'Wz via the sqrt-weight trick (memory O(n p), matching
-- solver.rs's "sqrt-weighted approach" comment).
def xtwx [n][p] (xw: [n][p]f32) : [p][p]f32 =
  map (\col_i -> map (dotprod col_i) (transpose xw)) (transpose xw)

def hadamard_sum [p] (a: [p][p]f32) (b: [p][p]f32) : f32 =
  reduce (+) 0 (map2 dotprod a b)

def quadform [p] (s: [p][p]f32) (beta: [p]f32) : f32 =
  dotprod beta (matvec s beta)

def reml_grad_eval [n][p][k]
    (x: [n][p]f32)        -- model matrix
    (z: [n]f32)           -- working response
    (w: [n]f32)           -- working weights (diagonal of W)
    (s: [k][p][p]f32)     -- penalty matrices S_j
    (beta: [p]f32)        -- current PWLS coefficients
    (v: [p][p]f32)        -- (X'WX + S_lambda)^-1 (from the CPU solve)
    (s_pinv: [p][p]f32)   -- Moore-Penrose pseudo-inverse of S_lambda
    (lambdas: [k]f32)     -- current smoothing parameters
    : ([p][p]f32, [p]f32, f32, [p]f32, [k]f32) =
  let sqrt_w = map f32.sqrt w
  let xw = map2 (\swi row -> map (* swi) row) sqrt_w x
  let zw = map2 (*) sqrt_w z
  -- X'WX (the dominant matmul) and X'Wz.
  let xtwx_m = xtwx xw
  let xtwz = map (`dotprod` zw) (transpose xw)
  -- Residual pipeline.
  let fitted = matvec x beta
  let r = map2 (-) z fitted
  let rss = reduce (+) 0 (map2 (*) w (map (\ri -> ri * ri) r))
  let rw = map2 (*) sqrt_w r
  let xtwr = map (`dotprod` rw) (transpose xw)
  -- Per-penalty REML gradient (solver.rs:258-268).
  let grad =
    map2 (\s_j lambda_j ->
            let bsb = quadform s_j beta
            let tr_v = hadamard_sum v s_j
            let tr_pinv = hadamard_sum s_pinv s_j
            in -(0.5 * lambda_j * (-bsb + tr_pinv - tr_v)))
         s lambdas
  in (xtwx_m, xtwz, rss, xtwr, grad)

-- Default entry so `futhark bench kernel.fut` with no entry filter
-- still works; delegates to the real thing.
-- ==
-- entry: reml_grad_eval_scaled
-- compiled random input { [2000][32]f32 [2000]f32 [2000]f32 [2][32][32]f32 [32]f32 [32][32]f32 [32][32]f32 [2]f32 }
-- auto output
entry reml_grad_eval_scaled [n][p][k]
    (x: [n][p]f32) (z: [n]f32) (w: [n]f32) (s: [k][p][p]f32)
    (beta: [p]f32) (v: [p][p]f32) (s_pinv: [p][p]f32) (lambdas: [k]f32)
    : ([p][p]f32, [p]f32, f32, [p]f32, [k]f32) =
  let (xtwx_m, xtwz, rss, xtwr, grad) =
    reml_grad_eval x z w s beta v s_pinv lambdas
  let c = 1 / f32.i64 n
  in (map (map (* c)) xtwx_m, map (* c) xtwz, c * rss,
      map (* c) xtwr, map (* c) grad)

-- ==
-- entry: main
-- input { [[1.0f32,0.0f32],[0.0f32,1.0f32],[1.0f32,1.0f32]] [1.0f32,2.0f32,3.0f32]
--         [1.0f32,1.0f32,1.0f32] [[[1.0f32,0.0f32],[0.0f32,1.0f32]]] [1.0f32,2.0f32]
--         [[1.0f32,0.0f32],[0.0f32,1.0f32]] [[1.0f32,0.0f32],[0.0f32,1.0f32]] [2.0f32] }
-- output { 5.0f32 }
entry main [n][p][k] (x: [n][p]f32) (z: [n]f32) (w: [n]f32)
                     (s: [k][p][p]f32) (beta: [p]f32)
                     (v: [p][p]f32) (s_pinv: [p][p]f32)
                     (lambdas: [k]f32) : f32 =
  let (_, _, rss, _, grad) = reml_grad_eval x z w s beta v s_pinv lambdas
  in rss + reduce (+) 0 grad
