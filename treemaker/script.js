let nodeId = 1;
const tree = { root: { feature: 'root', children: [] } };
const treeData = [{ id: 'root', parent: null, data: { feature: 'root' } }];

document.getElementById('add-root-btn').addEventListener('click', addRootNode);
document.getElementById('add-node-btn').addEventListener('click', addNode);
document.getElementById('download-btn').addEventListener('click', downloadJSON);

function addRootNode() {
    const container = document.getElementById('tree-container');
    const rootElement = document.createElement('div');
    rootElement.className = 'node';
    rootElement.id = 'node-root';
    rootElement.innerHTML = `
        <strong>Root Node</strong>
        <label for="feature-root">Feature:</label>
        <input type="text" id="feature-root" value="root" disabled>
        <button onclick="addSplit('root')">Add Split</button>
    `;
    container.appendChild(rootElement);
    visualizeTree();
    document.getElementById('add-root-btn').disabled = true;
    document.getElementById('add-node-btn').disabled = false;
    document.getElementById('download-btn').disabled = false;
}

function addNode() {
    const parentId = prompt("Enter parent node ID (e.g., 'root' for the root node):");
    if (!tree[parentId]) {
        alert("Parent node ID not found!");
        return;
    }
    addSplit(parentId);
}

function addSplit(parentId) {
    const parentNode = document.getElementById(`node-${parentId}`);
    if (!parentNode) {
        alert("Parent node element not found!");
        return;
    }

    const leftChildId = `node-${nodeId++}`;
    const rightChildId = `node-${nodeId++}`;

    const leftChildElement = document.createElement('div');
    leftChildElement.className = 'node';
    leftChildElement.id = leftChildId;
    leftChildElement.innerHTML = `
        <label for="feature-${leftChildId}">Feature:</label>
        <input type="text" id="feature-${leftChildId}">
        <label for="condition-${leftChildId}">Condition:</label>
        <select id="condition-${leftChildId}">
            <option value="less_than">less than</option>
            <option value="greater_than">greater than</option>
        </select>
        <label for="value-${leftChildId}">Value:</label>
        <input type="number" id="value-${leftChildId}">
        <button onclick="saveNode('${leftChildId}', '${parentId}', 'left')">Save Node</button>
        <button onclick="addSplit('${leftChildId}')">Add Split</button>
    `;

    const rightChildElement = document.createElement('div');
    rightChildElement.className = 'node';
    rightChildElement.id = rightChildId;
    rightChildElement.innerHTML = `
        <label for="feature-${rightChildId}">Feature:</label>
        <input type="text" id="feature-${rightChildId}">
        <label for="condition-${rightChildId}">Condition:</label>
        <select id="condition-${rightChildId}">
            <option value="less_than">less than</option>
            <option value="greater_than">greater than</option>
        </select>
        <label for="value-${rightChildId}">Value:</label>
        <input type="number" id="value-${rightChildId}">
        <button onclick="saveNode('${rightChildId}', '${parentId}', 'right')">Save Node</button>
        <button onclick="addSplit('${rightChildId}')">Add Split</button>
    `;

    const splitContainer = document.createElement('div');
    splitContainer.className = 'split-container';
    splitContainer.appendChild(leftChildElement);
    splitContainer.appendChild(rightChildElement);

    parentNode.appendChild(splitContainer);
}

function saveNode(id, parentId, side) {
    const feature = document.getElementById(`feature-${id}`).value;
    const condition = document.getElementById(`condition-${id}`).value;
    const value = parseFloat(document.getElementById(`value-${id}`).value);

    const nodeData = { feature, condition, value, children: [] };

    if (!tree[parentId].children) {
        tree[parentId].children = [];
    }
    tree[parentId].children.push(nodeData);

    tree[id] = nodeData;

    updateTreeData();
    visualizeTree();
}

function updateTreeData() {
    treeData.length = 1; // Reset tree data but keep the root
    let nodeIdCounter = 1;
    function traverse(node, parentId) {
        const nodeId = `split${nodeIdCounter++}`;
        treeData.push({
            id: nodeId,
            parent: parentId,
            data: node
        });
        node.children.forEach(child => traverse(child, nodeId));
    }
    traverse(tree.root, 'root');
}

function visualizeTree() {
    const svg = d3.select("svg");
    svg.selectAll("*").remove(); // Clear the existing tree

    const width = +svg.attr("width");
    const height = +svg.attr("height");

    const root = d3.stratify()
        .id(d => d.id)
        .parentId(d => d.parent)(treeData);

    const treeLayout = d3.tree().size([height, width - 160]);
    treeLayout(root);

    // Links
    svg.selectAll('path.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x))
        .attr('fill', 'none')
        .attr('stroke', '#ffffff');

    // Nodes
    const node = svg.selectAll('g.node')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', d => `translate(${d.y},${d.x})`);

    node.append('circle')
        .attr('r', 5)
        .attr('fill', '#ffffff');

    node.append('text')
        .attr('dy', '.35em')
        .attr('x', d => d.children ? -10 : 10)
        .style('text-anchor', d => d.children ? 'end' : 'start')
        .attr('fill', '#ffffff')
        .text(d => d.id);
}

function downloadJSON() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(tree, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "decision_tree.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}
