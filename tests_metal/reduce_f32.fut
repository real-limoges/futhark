-- f32 dot product: reduction with a fused map.
-- ==
-- input { [1.0f32, 2.0f32, 3.0f32] [4.0f32, 5.0f32, 6.0f32] }
-- output { 32.0f32 }

def main (xs: []f32) (ys: []f32) : f32 =
  reduce (+) 0 (map2 (*) xs ys)
