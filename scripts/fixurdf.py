import sys
import xml.etree.ElementTree as ET

MASS_EPS = 1e-4         
DIAG_FLOOR = 1e-11      

def f(x):
    return float(x)

def main(inp, outp):
    tree = ET.parse(inp)
    root = tree.getroot()

    changed = []

    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            continue

        m = f(mass.attrib.get("value", "0"))
        if m > MASS_EPS:
            continue

        ixx = f(inertia.attrib.get("ixx", "0"))
        iyy = f(inertia.attrib.get("iyy", "0"))
        izz = f(inertia.attrib.get("izz", "0"))
        ixy = f(inertia.attrib.get("ixy", "0"))
        ixz = f(inertia.attrib.get("ixz", "0"))
        iyz = f(inertia.attrib.get("iyz", "0"))

        if abs(ixy) > 0 or abs(ixz) > 0 or abs(iyz) > 0:
            inertia.attrib["ixy"] = "0"
            inertia.attrib["ixz"] = "0"
            inertia.attrib["iyz"] = "0"

            if ixx <= 0: inertia.attrib["ixx"] = str(DIAG_FLOOR)
            if iyy <= 0: inertia.attrib["iyy"] = str(DIAG_FLOOR)
            if izz <= 0: inertia.attrib["izz"] = str(DIAG_FLOOR)

            changed.append(name)

    tree.write(outp, encoding="UTF-8", xml_declaration=True)
    print(f"Patched {len(changed)} links:")
    for n in changed:
        print(" -", n)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_urdf_inertia.py input.urdf output.urdf")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
