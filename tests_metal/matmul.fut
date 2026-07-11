-- Matrix multiplication: exercises tiling, threadgroup memory, and
-- transposition (builtin kernels).
-- ==
-- input { [[1.0f32, 2.0f32], [3.0f32, 4.0f32]] [[5.0f32, 6.0f32], [7.0f32, 8.0f32]] }
-- output { [[19.0f32, 22.0f32], [43.0f32, 50.0f32]] }
-- compiled random input { [128][64]f32 [64][32]f32 } auto output

def main [n] [m] [p] (a: [n][m]f32) (b: [m][p]f32) : [n][p]f32 =
  map (\a_row ->
         map (\b_col -> reduce (+) 0 (map2 (*) a_row b_col))
             (transpose b))
      a
