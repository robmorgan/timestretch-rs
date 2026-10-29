import Cocoa
import CoreGraphics

func layer0Windows() -> [[String: Any]] {
    let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements],
                                          kCGNullWindowID) as! [[String: Any]]
    return list.filter { ($0[kCGWindowLayer as String] as? Int ?? -1) == 0 }
}

func bounds(_ w: [String: Any]) -> CGRect {
    let b = w[kCGWindowBounds as String] as! [String: CGFloat]
    return CGRect(x: b["X"]!, y: b["Y"]!, width: b["Width"]!, height: b["Height"]!)
}

func post(_ type: CGEventType, _ pt: CGPoint, _ button: CGMouseButton = .left) {
    CGEvent(mouseEventSource: nil, mouseType: type,
            mouseCursorPosition: pt, mouseButton: button)?.post(tap: .cghidEventTap)
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    fputs("usage: uidrive winid <name> | list | click <x> <y> | drag <x1> <y1> <x2> <y2> <ms> [holdBeforeMs] [holdAfterMs]\n", stderr)
    exit(2)
}

switch args[1] {
case "winid":
    let name = args[2].lowercased()
    for w in layer0Windows() {
        let owner = (w[kCGWindowOwnerName as String] as? String ?? "").lowercased()
        if owner.contains(name) {
            let id = w[kCGWindowNumber as String] as! Int
            let r = bounds(w)
            print("\(id) \(Int(r.origin.x)) \(Int(r.origin.y)) \(Int(r.width)) \(Int(r.height))")
            exit(0)
        }
    }
    fputs("window not found\n", stderr)
    exit(1)

case "list":
    for w in layer0Windows() {
        let owner = w[kCGWindowOwnerName as String] as? String ?? "?"
        let title = w[kCGWindowName as String] as? String ?? ""
        let id = w[kCGWindowNumber as String] as! Int
        let r = bounds(w)
        print("\(id)\t\(owner)\t\(title)\t\(Int(r.origin.x)) \(Int(r.origin.y)) \(Int(r.width)) \(Int(r.height))")
    }

case "click":
    let pt = CGPoint(x: Double(args[2])!, y: Double(args[3])!)
    post(.mouseMoved, pt)
    usleep(50_000)
    post(.leftMouseDown, pt)
    usleep(60_000)
    post(.leftMouseUp, pt)

case "drag":
    let p1 = CGPoint(x: Double(args[2])!, y: Double(args[3])!)
    let p2 = CGPoint(x: Double(args[4])!, y: Double(args[5])!)
    let ms = Double(args[6])!
    let holdBefore = args.count > 7 ? Double(args[7])! : 0
    let holdAfter = args.count > 8 ? Double(args[8])! : 0
    post(.mouseMoved, p1)
    usleep(50_000)
    post(.leftMouseDown, p1)
    if holdBefore > 0 { usleep(useconds_t(holdBefore * 1000)) }
    let steps = max(Int(ms / 8), 2)
    for i in 1...steps {
        let t = Double(i) / Double(steps)
        let pt = CGPoint(x: p1.x + (p2.x - p1.x) * t, y: p1.y + (p2.y - p1.y) * t)
        post(.leftMouseDragged, pt)
        usleep(useconds_t((ms / Double(steps)) * 1000))
    }
    if holdAfter > 0 { usleep(useconds_t(holdAfter * 1000)) }
    post(.leftMouseUp, p2)

default:
    fputs("unknown subcommand \(args[1])\n", stderr)
    exit(2)
}
