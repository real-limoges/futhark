-- Segmented scan (regular segments), the differentiator construct.
-- ==
-- input { [[1i32, 2i32, 3i32], [4i32, 5i32, 6i32]] }
-- output { [[1i32, 3i32, 6i32], [4i32, 9i32, 15i32]] }
-- compiled random input { [1000][512]i32 } auto output

def main (xss: [][]i32) : [][]i32 = map (scan (+) 0) xss
