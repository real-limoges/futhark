{-# LANGUAGE TemplateHaskell #-}

-- | Code snippets and support files used by the Metal backend.
module Futhark.CodeGen.RTS.Metal
  ( preludeMetal,
    shimCppMetal,
    metalCppFiles,
  )
where

import Data.ByteString qualified as BS
import Data.FileEmbed
import Data.Text qualified as T

-- | @rts/metal/prelude.metal@
preludeMetal :: T.Text
preludeMetal = $(embedStringFile "rts/metal/prelude.metal")
{-# NOINLINE preludeMetal #-}

-- | @rts/metal/shim.cpp@ — the metal-cpp translation unit that is
-- compiled and linked alongside the generated C program.
shimCppMetal :: T.Text
shimCppMetal = $(embedStringFile "rts/metal/shim.cpp")
{-# NOINLINE shimCppMetal #-}

-- | The vendored metal-cpp headers, materialised to a cache directory
-- when compiling generated programs.
metalCppFiles :: [(FilePath, BS.ByteString)]
metalCppFiles = $(embedDir "rts/metal/metal-cpp")
{-# NOINLINE metalCppFiles #-}
