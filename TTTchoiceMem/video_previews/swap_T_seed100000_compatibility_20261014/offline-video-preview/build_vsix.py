"""Package the small dependency-free, local extension without touching editor files."""

import json
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    source_directory = Path(__file__).resolve().parent
    manifest = json.loads((source_directory / "package.json").read_text())
    destination = source_directory.parent / "memory-local.offline-video-preview-0.0.1.vsix"
    extension_manifest = f'''<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id="{manifest['name']}" Version="{manifest['version']}" Publisher="{manifest['publisher']}" />
    <DisplayName>{escape(manifest['displayName'])}</DisplayName>
    <Description xml:space="preserve">{escape(manifest['description'])}</Description>
    <Categories>Visualization</Categories><Tags>video,offline</Tags>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="^1.90.0" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="workspace" />
    </Properties>
  </Metadata>
  <Installation><InstallationTarget Id="Microsoft.VisualStudio.Code" /></Installation>
  <Dependencies />
  <Assets><Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" /></Assets>
</PackageManifest>'''
    content_types = '''<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="json" ContentType="application/json" />
  <Default Extension="js" ContentType="application/javascript" />
  <Default Extension="vsixmanifest" ContentType="text/xml" />
</Types>'''
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as package:
        package.writestr("extension.vsixmanifest", extension_manifest)
        package.writestr("[Content_Types].xml", content_types)
        for file_name in ("package.json", "extension.js", "playerHtml.js"):
            package.write(source_directory / file_name, f"extension/{file_name}")
    print(destination)


if __name__ == "__main__":
    main()
