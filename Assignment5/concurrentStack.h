class Node {
public:
    int value;
    Node* next;
    Node(int v) : value(v), next(nullptr) {}
};

struct TopWithCount {
    Node* node;  
    uint32_t count;
};

class LockFreeStack {
private:
    std::atomic<uintptr_t> top_with_count; // Atomic combined pointer and count

public:
    LockFreeStack() : top_with_count(0) {}

    void push(int value) {
        auto new_node = new Node(value);
        uintptr_t expected = top_with_count.load(); // Load the initial top state

        while (true) {
            TopWithCount old_top = unpack(expected);

            new_node->next = old_top.node;
            uintptr_t desired = pack(new_node, old_top.count);

            if (top_with_count.compare_exchange_weak(expected, desired)) {
                return;
            }
        }
    }

    int pop() {
        uintptr_t expected = top_with_count.load();

        while (true) {
            TopWithCount old_top = unpack(expected);
            if (!old_top.node) {
                return -1;
            }
            TopWithCount new_top = {old_top.node->next, old_top.count + 1};
            uintptr_t desired = pack(new_top.node, new_top.count);

            if (top_with_count.compare_exchange_weak(expected, desired)) {
                int value = old_top.node->value;
                // delete old_top.node;
                return value;
            }
        }
    }


    
    void printStack() { 
        TopWithCount current = unpack(top_with_count.load()); 
        Node* current_node = current.node;

        std::cout << "Stack contents (from top to bottom): ";
        if (!current_node) {
            std::cout << "Stack is empty.\n";
            return;
        }
        while (current_node != nullptr) {
            if (current_node) {
                std::cout << current_node->value << " ";
                current_node = current_node->next;
            } 
            else {
                std::cerr << "Error: encountered a null node during traversal.\n";
                break;  // Prevent segfault in case of a corrupted stack
            }
        }
        std::cout << "\n";
    }

private:
    uintptr_t pack(Node* node, uint32_t count) {
        return reinterpret_cast<uintptr_t>(node) | (static_cast<uintptr_t>(count) << 48);
    }

    TopWithCount unpack(uintptr_t combined) {
        Node* node = reinterpret_cast<Node*>(combined & ((1ULL << 48) - 1));
        uint32_t count = static_cast<uint32_t>(combined >> 48);
        return {node, count};
    }
};