#!/usr/bin/env bash
# Deploy the BESS optimiser demo to AWS ECS Fargate.
#
# Prereqs (one-time): AWS CLI configured, Docker running, an ecsTaskExecutionRole,
# a default VPC with a public subnet, and the OpenAI key stored in SSM:
#   aws ssm put-parameter --name /bess/openai_api_key --type SecureString --value sk-...
#
# This script: builds + pushes the image to ECR, registers the task definition,
# and runs it as a Fargate service with a public IP. Idempotent-ish; re-run to
# ship a new image (forces a new deployment).
#
# Cost note: a single 1 vCPU / 4 GB Fargate task runs ~A$50-60/month if left on.
# Stop it with:  aws ecs update-service --cluster bess --service bess-svc --desired-count 0
set -euo pipefail

REGION="${AWS_REGION:-ap-southeast-2}"        # Sydney
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO="bess-optimiser"
CLUSTER="bess"
SERVICE="bess-svc"
IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"

echo "==> Account ${ACCOUNT_ID}  Region ${REGION}"

# 1. ECR repo + login + build + push
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker build -t "$REPO:latest" ..
docker tag "$REPO:latest" "$IMAGE"
docker push "$IMAGE"

# 2. Register task definition (substitute placeholders)
TMP="$(mktemp)"
sed -e "s/ACCOUNT_ID/${ACCOUNT_ID}/g" -e "s/REGION/${REGION}/g" ecs-task-def.json > "$TMP"
TASK_ARN="$(aws ecs register-task-definition --cli-input-json "file://${TMP}" \
  --region "$REGION" --query 'taskDefinition.taskDefinitionArn' --output text)"
echo "==> Registered ${TASK_ARN}"

# 3. Cluster + networking (uses default VPC's first public subnet)
aws ecs describe-clusters --clusters "$CLUSTER" --region "$REGION" \
  --query 'clusters[0].status' --output text 2>/dev/null | grep -q ACTIVE \
  || aws ecs create-cluster --cluster-name "$CLUSTER" --region "$REGION" >/dev/null

VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")"
SUBNET_ID="$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID \
  Name=map-public-ip-on-launch,Values=true \
  --query 'Subnets[0].SubnetId' --output text --region "$REGION")"

SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values=bess-sg \
  Name=vpc-id,Values=$VPC_ID --query 'SecurityGroups[0].GroupId' --output text \
  --region "$REGION" 2>/dev/null || true)"
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID="$(aws ec2 create-security-group --group-name bess-sg \
    --description "BESS demo" --vpc-id "$VPC_ID" --region "$REGION" \
    --query GroupId --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 8501 --cidr 0.0.0.0/0 --region "$REGION" >/dev/null
fi

NET="awsvpcConfiguration={subnets=[${SUBNET_ID}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}"

# 4. Create or update the service
if aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
     --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
  aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" \
    --task-definition "$TASK_ARN" --force-new-deployment --region "$REGION" >/dev/null
  echo "==> Updated service (new deployment rolling out)"
else
  aws ecs create-service --cluster "$CLUSTER" --service-name "$SERVICE" \
    --task-definition "$TASK_ARN" --desired-count 1 --launch-type FARGATE \
    --network-configuration "$NET" --region "$REGION" >/dev/null
  echo "==> Created service"
fi

echo "==> Done. Find the public IP:"
echo "    TASK=\$(aws ecs list-tasks --cluster $CLUSTER --service-name $SERVICE --region $REGION --query 'taskArns[0]' --output text)"
echo "    ENI=\$(aws ecs describe-tasks --cluster $CLUSTER --tasks \$TASK --region $REGION --query 'tasks[0].attachments[0].details[?name==\`networkInterfaceId\`].value' --output text)"
echo "    aws ec2 describe-network-interfaces --network-interface-ids \$ENI --region $REGION --query 'NetworkInterfaces[0].Association.PublicIp' --output text"
echo "    open http://<public-ip>:8501"
