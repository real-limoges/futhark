-- Simple map, the first Metal smoke test.
-- ==
-- input { [1i32, 2i32, 3i32, 4i32, 5i32] }
-- output { [3i32, 4i32, 5i32, 6i32, 7i32] }
-- input { empty([0]i32) }
-- output { empty([0]i32) }

def main (xs: []i32) : []i32 = map (+ 2) xs
